from sqlalchemy import inspect, text


def add_columns(engine, table, definitions):
    """Additive migrations for the prototype database; no rows are discarded."""
    existing = {column["name"] for column in inspect(engine).get_columns(table)}
    with engine.begin() as connection:
        for name, definition in definitions.items():
            if name not in existing:
                # Names and definitions are constants supplied by our code, never user input.
                connection.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {definition}'))
