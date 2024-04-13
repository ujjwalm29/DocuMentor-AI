from dataclasses import fields


def get_dataclass_fields(data_cls):
    return {field.name: field.type for field in fields(data_cls)}