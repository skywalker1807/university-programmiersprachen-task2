#!/usr/bin/env python3
import argparse
from pathlib import Path
from enum import StrEnum, auto

# integers
# structured data
# functions
# predefined operations

# Tokens
# integer-literals
# names
# symbols
# parenthesis

class TokenType(StrEnum):
    identifier = auto() # [a-zA-Z]+
    integer = auto() # [0-9]+

    # syntactic elements
    equal_sign = auto()
    comma = auto()
    dot = auto()

    # parenthesis
    left_parenthesis = auto()
    right_parenthesis = auto()
    left_bracket = auto()
    right_bracket = auto()
    left_brace = auto()
    right_brace = auto()

    # predefined functions
    plus = auto()
    minus = auto()
    mult = auto()
    dev = auto()
    cond = auto()

    # whitespace
    whitespace = auto()
    new_line = auto()

class SourceCodeLocation:

    def __init__(self, line_index: int, start_index: int, end_index: int) -> None:
        self._line_index = line_index
        self._start_index = start_index
        self._end_index = end_index

    @property
    def line_index(self) -> int:
        return self._line_index

    @property
    def start_index(self) -> int:
        return self._line_index

    @property
    def end_index(self) -> int:
        return self._line_index

    def __str__(self) -> str:
        return f"SourceCodeLocation{{line_index={self._line_index}, start_index={self._start_index}, end_index={self._end_index}}}"

    def str_short(self) -> str:
        return f"{self._line_index}:{self._start_index}-{self._end_index}"

class Token:
    def __init__(self, token_type: TokenType, source_code_location: SourceCodeLocation) -> None:
        self._type = token_type
        self._location = source_code_location

    @property
    def token_type(self) -> TokenType:
        return self._type

    @property
    def source_code_location() -> SourceCodeLocation:
        return self._location

    def __str__(self) -> str:
        return f"Token{{token_type={self._type}, location={self._location.str_short()}}}"

def group_characters(source_code: str) -> list[Token]:
    start_index = 0
    end_index = 0
    peek_index = 0

    line_index = 1
    character_index = 1

    tokens: list[Token] = []

    while end_index < len(source_code):
        end_index += 1

        match(source_code[start_index:end_index]):
            case '\n':
                tokens.append(Token(
                    token_type=TokenType.new_line,
                    source_code_location=SourceCodeLocation(
                        line_index=line_index,
                        start_index=character_index,
                        end_index=character_index + 1)
                ))
                line_index += 1
                character_index = 1

                start_index = end_index
                end_index = start_index

            case ' ':
                # peek all whitespace
                peek_index = 1
                while end_index + peek_index < len(source_code):
                    if source_code[end_index + peek_index] != ' ':
                        break
                    peek_index += 1

                tokens.append(Token(
                    token_type=TokenType.whitespace,
                    source_code_location=SourceCodeLocation(
                        line_index=line_index,
                        start_index=character_index,
                        end_index=character_index + peek_index)
                ))
                character_index += peek_index

                start_index = end_index + peek_index
                end_index = start_index

            case _:
                character_index += 1
                print(source_code[start_index:end_index], end='', flush=True)

                start_index = end_index
                end_index = start_index

    return tokens

def read_source_code(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def main() -> None:
    parser = argparse.ArgumentParser(description="f1 intperpreter")
    parser.add_argument("filename", help="Path to the file to interpret")
    args = parser.parse_args()

    source_code = read_source_code(args.filename)

    tokens = group_characters(source_code)

    for token in tokens:
        print(token)

if __name__ == '__main__':
    main()
