from concurrent.futures import ThreadPoolExecutor

from services.rag.catalog import Catalog


def test_import_is_durable_and_invisible_until_completed(tmp_path):
    catalog = Catalog(tmp_path, "embeddinggemma")
    catalog.enqueue("abc", "paper.pdf", b"%PDF-test")
    assert catalog.ready_ids() == [] and catalog.revision() == 0
    claimed = catalog.claim()
    catalog.progress("abc", 8, 20, 3)
    assert claimed["status"] == "processing" and catalog.ready_ids() == []
    catalog.engine.dispose()
    restored = Catalog(tmp_path, "embeddinggemma")
    assert restored.documents()[0]["status"] == "queued"
    assert restored.path("abc").read_bytes() == b"%PDF-test"
    restored.claim()
    restored.progress("abc", 20, 20, 3)
    restored.complete("abc")
    assert restored.ready_ids() == ["abc"] and restored.revision() == 1
    restored.enqueue("abc", "different-name.pdf", b"%PDF-test")
    assert len(restored.documents()) == 1 and restored.revision() == 1
    restored.remove("abc")
    assert restored.ready_ids() == [] and restored.revision() == 2
    assert restored.path("abc").exists()  # Logical removal is recoverable.
    restored.engine.dispose()


def test_parallel_import_claims_are_unique_and_revisions_do_not_get_lost(tmp_path):
    catalog = Catalog(tmp_path, "embeddinggemma")
    for index in range(10):
        catalog.enqueue(str(index), f"{index}.pdf", b"%PDF-test")
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: catalog.claim(), range(10)))
    assert len({doc["id"] for doc in claims}) == 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda doc: catalog.complete(doc["id"]), claims))
    assert catalog.revision() == 10 and len(catalog.ready_ids()) == 10
    catalog.engine.dispose()


def test_failed_import_can_be_retried_without_new_document_id(tmp_path):
    catalog = Catalog(tmp_path, "embeddinggemma")
    catalog.enqueue("abc", "paper.pdf", b"%PDF-test")
    catalog.claim()
    catalog.fail("abc", "bad pdf", 3, permanent=True)
    assert catalog.documents()[0]["status"] == "failed"
    catalog.retry("abc")
    assert catalog.documents()[0]["status"] == "queued"
    assert len(catalog.documents()) == 1
    catalog.engine.dispose()
