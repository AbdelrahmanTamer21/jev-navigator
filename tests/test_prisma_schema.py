"""Prisma schema blocks: each model, view, enum and composite type of a ``.prisma`` file, its lines,
and the client accessor a model is queried through.

The main test reads umami's real schema (``tests/fixtures/umami``); the small schemas below each
hold one shape a line-by-line scanner gets wrong.
"""

from __future__ import annotations

from jev_navigator.index.languages import split_lines
from jev_navigator.index.prisma_schema import SchemaBlock, is_schema_file, schema_blocks

EVERY_KIND = """\
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id    Int    @id
  email String @unique
  role  Role
}

enum Role {
  USER
  ADMIN
}

view UserInfo {
  id    Int    @unique
  email String
}

type Address {
  street String
  city   String
}
"""

BRACES_IN_STRINGS_AND_COMMENTS = """\
model Run {
  id     String @id
  config Json   @default("{}")
  // A config object ends with } like any JSON object
  label  String @map("run}")
}

enum Status {
  OPEN
  RESOLVED
}
"""

UNCLOSED_THEN_CLOSED = """\
model Broken {
  id Int @id

model Next {
  id Int @id
}
"""


def test_each_model_of_a_real_schema_is_one_block_from_its_header_to_its_closing_brace(
    umami_schema: str,
) -> None:
    # Act
    blocks = schema_blocks(split_lines(umami_schema))

    # Assert: 26 models; the generator and datasource settings are no blocks.
    assert len(blocks) == 26
    assert {block.keyword for block in blocks} == {"model"}
    assert blocks[0] == SchemaBlock("model", "User", 12, 38)
    assert SchemaBlock("model", "Website", 98, 131) in blocks
    assert blocks[-1] == SchemaBlock("model", "AppSetting", 529, 534)
    assert all(umami_schema.split("\n")[block.end - 1] == "}" for block in blocks)


def test_a_models_client_accessor_is_its_name_with_the_first_character_lower_cased(umami_schema: str) -> None:
    # Act
    accessors = {block.name: block.client_accessor for block in schema_blocks(split_lines(umami_schema))}

    # Assert: Prisma Client spells `model WebsiteEvent` as `prisma.websiteEvent`, and lower-cases only
    # the first character, so `model URL` is `prisma.uRL`.
    assert accessors["Website"] == "website"
    assert accessors["WebsiteEvent"] == "websiteEvent"
    assert accessors["TwoFactorOtpUsed"] == "twoFactorOtpUsed"
    assert SchemaBlock("model", "URL", 1, 3).client_accessor == "uRL"


def test_enums_views_and_composite_types_are_blocks_and_only_models_and_views_are_queried() -> None:
    # Act
    blocks = schema_blocks(split_lines(EVERY_KIND))

    # Assert
    assert [
        (block.keyword, block.name, block.start, block.end, block.client_accessor) for block in blocks
    ] == [
        ("model", "User", 10, 14, "user"),
        ("enum", "Role", 16, 19, None),
        ("view", "UserInfo", 21, 24, "userInfo"),
        ("type", "Address", 26, 29, None),
    ]


def test_braces_in_strings_and_comments_never_open_or_close_a_block() -> None:
    # Act
    blocks = schema_blocks(split_lines(BRACES_IN_STRINGS_AND_COMMENTS))

    # Assert
    assert [(block.name, block.start, block.end) for block in blocks] == [("Run", 1, 6), ("Status", 8, 11)]


def test_a_header_without_its_closing_brace_is_no_block_and_the_blocks_after_it_still_are() -> None:
    # Act
    blocks = schema_blocks(split_lines(UNCLOSED_THEN_CLOSED))

    # Assert
    assert blocks == (SchemaBlock("model", "Next", 4, 6),)


def test_a_schema_file_is_named_by_its_prisma_suffix() -> None:
    # Assert
    assert is_schema_file("prisma/schema.prisma")
    assert is_schema_file("prisma/schema/website.prisma")
    assert not is_schema_file("src/lib/prisma.ts")
    assert not is_schema_file("prisma/.prisma")
