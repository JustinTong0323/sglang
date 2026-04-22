"""Unit tests for tool-parameter schema alias normalization."""

import unittest

from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.function_call.utils import normalize_json_schema_types
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestNormalizeJsonSchemaTypes(CustomTestCase):
    def _assert_accepts(self, schema: dict) -> None:
        Draft202012Validator.check_schema(schema)

    def test_enum_alias_becomes_string(self):
        schema = {
            "type": "object",
            "properties": {"color": {"type": "enum", "enum": ["red", "green", "blue"]}},
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["color"]["type"], "string")
        self.assertEqual(
            schema["properties"]["color"]["enum"], ["red", "green", "blue"]
        )
        self._assert_accepts(schema)

    def test_varchar_alias_becomes_string(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "varchar"},
                "short_name": {"type": "VARCHAR(255)"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertEqual(schema["properties"]["short_name"]["type"], "string")
        self._assert_accepts(schema)

    def test_numeric_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "int"},
                "big": {"type": "bigint"},
                "price": {"type": "decimal(10,2)"},
                "ratio": {"type": "float"},
            },
        }
        normalize_json_schema_types(schema)
        props = schema["properties"]
        self.assertEqual(props["age"]["type"], "integer")
        self.assertEqual(props["big"]["type"], "integer")
        self.assertEqual(props["price"]["type"], "number")
        self.assertEqual(props["ratio"]["type"], "number")
        self._assert_accepts(schema)

    def test_prefix_matched_numeric_types(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "int32"},
                "b": {"type": "int64"},
                "c": {"type": "uint"},
                "d": {"type": "unsigned"},
                "e": {"type": "long"},
                "f": {"type": "short"},
                "g": {"type": "float32"},
                "h": {"type": "float64"},
                "i": {"type": "num"},
                "j": {"type": "numeric"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        for k in ("a", "b", "c", "d", "e", "f"):
            self.assertEqual(p[k]["type"], "integer")
        for k in ("g", "h", "i", "j"):
            self.assertEqual(p[k]["type"], "number")
        self._assert_accepts(schema)

    def test_prefix_matched_compound_types(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "list[str]"},
                "b": {"type": "list_of_int"},
                "c": {"type": "dict"},
                "d": {"type": "dict[str, int]"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        self.assertEqual(p["a"]["type"], "array")
        self.assertEqual(p["b"]["type"], "array")
        self.assertEqual(p["c"]["type"], "object")
        self.assertEqual(p["d"]["type"], "object")
        self._assert_accepts(schema)

    def test_binary_and_arr_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "flag": {"type": "binary"},
                "items": {"type": "arr"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["flag"]["type"], "boolean")
        self.assertEqual(schema["properties"]["items"]["type"], "array")
        self._assert_accepts(schema)

    def test_case_insensitive(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "VARCHAR"},
                "b": {"type": "INT"},
                "c": {"type": "String"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        self.assertEqual(p["a"]["type"], "string")
        self.assertEqual(p["b"]["type"], "integer")
        self.assertEqual(p["c"]["type"], "string")
        self._assert_accepts(schema)

    def test_array_and_object_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "list", "items": {"type": "str"}},
                "meta": {"type": "dict"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["tags"]["type"], "array")
        self.assertEqual(schema["properties"]["tags"]["items"]["type"], "string")
        self.assertEqual(schema["properties"]["meta"]["type"], "object")
        self._assert_accepts(schema)

    def test_nested_anyof_and_defs(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "int"},
                        {"type": "varchar"},
                    ]
                }
            },
            "$defs": {
                "Row": {"type": "object", "properties": {"id": {"type": "bigint"}}}
            },
        }
        normalize_json_schema_types(schema)
        any_of = schema["properties"]["value"]["anyOf"]
        self.assertEqual(any_of[0]["type"], "integer")
        self.assertEqual(any_of[1]["type"], "string")
        self.assertEqual(schema["$defs"]["Row"]["properties"]["id"]["type"], "integer")
        self._assert_accepts(schema)

    def test_type_list_member_normalized(self):
        schema = {"type": ["varchar", "null"]}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["type"], ["string", "null"])
        self._assert_accepts(schema)

    def test_standard_types_untouched(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["a"]["type"], "string")
        self.assertEqual(schema["properties"]["b"]["type"], "integer")
        self.assertEqual(schema["properties"]["c"]["type"], "boolean")

    def test_unknown_type_left_alone(self):
        schema = {"type": "object", "properties": {"x": {"type": "geometry"}}}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["x"]["type"], "geometry")
        with self.assertRaises(SchemaError):
            self._assert_accepts(schema)

    def test_common_db_orm_type_names_accepted(self):
        """Common non-standard DB/ORM type names all survive validation."""
        recognized = [
            # string family
            "string",
            "str",
            "text",
            "varchar",
            "char",
            "enum",
            # integer via prefix
            "int",
            "int32",
            "int64",
            "uint",
            "uint8",
            "long",
            "long long",
            "short",
            "unsigned",
            # number via prefix
            "num",
            "numeric",
            "float",
            "float32",
            "float64",
            # boolean
            "boolean",
            "bool",
            "binary",
            # compound
            "object",
            "array",
            "arr",
            "dict",
            "dict[str, int]",
            "list",
            "list[str]",
        ]
        for t in recognized:
            schema = {"type": "object", "properties": {"x": {"type": t}}}
            normalize_json_schema_types(schema)
            try:
                self._assert_accepts(schema)
            except SchemaError as e:
                self.fail(f"type {t!r} → {schema['properties']['x']['type']!r}: {e}")

    def test_pre_existing_400_schema_now_accepted(self):
        schema = {
            "type": "object",
            "properties": {
                "sql": {"type": "varchar"},
                "mode": {"type": "enum", "enum": ["read", "write"]},
            },
            "required": ["sql", "mode"],
        }
        normalize_json_schema_types(schema)
        self._assert_accepts(schema)


if __name__ == "__main__":
    unittest.main()
