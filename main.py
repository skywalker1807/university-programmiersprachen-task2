#!/usr/bin/python3.12
import argparse
import sys
from pathlib import Path
from enum import StrEnum, auto
from copy import deepcopy, copy
import textwrap
import json


class SourceCode:
    def __init__(self, path: Path, content: str) -> None:
        self._path = path
        self._content = content

    @property
    def path(self) -> Path:
        return self._path

    @property
    def content(self) -> str:
        return self._content


def read_source_code(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


class TokenType(StrEnum):
    integer = auto()
    identifier = auto()

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

    # whitespace
    whitespace = auto()
    new_line = auto()


class SourceCodeLocation:
    def __init__(self, line_index: int, column_index: int, start_index: int, end_index: int) -> None:
        self.line_index = line_index
        self.column_index = column_index
        self.start_index = start_index
        self.end_index = end_index

    def __str__(self) -> str:
        length = self.end_index - self.start_index
        return f"{self.line_index}:{self.column_index}-{self.column_index + length}"

    def advance(self) -> None:
        length = self.end_index - self.start_index
        self.column_index += length

        self.start_index = self.end_index
        self.end_index = self.start_index

    def advance_line(self) -> None:
        self.line_index += 1
        self.column_index = 1

        self.start_index = self.end_index
        self.end_index = self.start_index


class Token:
    def __init__(self, token_type: TokenType, source_code: SourceCode, source_code_location: SourceCodeLocation) -> None:
        self._token_type = token_type
        self._source_code = source_code
        self._source_code_location = source_code_location

    @property
    def token_type(self) -> TokenType:
        return self._token_type

    @property
    def literal(self) -> str:
        return self._source_code.content[self._source_code_location.start_index: self._source_code_location.end_index]

    @property
    def source_code_location(self) -> SourceCodeLocation:
        return self._source_code_location

    @property
    def source_code(self) -> SourceCode:
        return self._source_code

    def __str__(self) -> str:
        return f"Token{{literal='{self.literal}', token_type={self._token_type}, location={self._source_code_location}}}"


def is_part_of_identifier(x: str) -> bool:
    return (x.isascii() and x.isalpha()) or x == "_"


class InvalidSourceCodeCharacter(Exception):
    pass


def group_characters(source_code: SourceCode) -> list[Token] | InvalidSourceCodeCharacter:
    location = SourceCodeLocation(
        line_index=1,
        column_index=1,
        start_index=0,
        end_index=0,
    )

    tokens: list[Token] = []

    while location.end_index < len(source_code.content):
        location.end_index += 1

        match source_code.content[location.start_index: location.end_index]:
            case '-':
                # peek all digits
                peek_index = 0
                while location.end_index + peek_index < len(source_code.content):
                    if not source_code.content[location.end_index + peek_index].isdigit():
                        break
                    peek_index += 1

                location.end_index += peek_index
                tokens.append(
                    Token(
                        token_type=TokenType.integer,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case x if x.isdigit():
                # peek all digits
                peek_index = 0
                while location.end_index + peek_index < len(source_code.content):
                    if not source_code.content[location.end_index + peek_index].isdigit():
                        break
                    peek_index += 1

                location.end_index += peek_index
                tokens.append(
                    Token(
                        token_type=TokenType.integer,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case x if is_part_of_identifier(x):
                # peek all identifier characters
                peek_index = 0
                while location.end_index + peek_index < len(source_code.content):
                    if not is_part_of_identifier(source_code.content[location.end_index + peek_index]):
                        break
                    peek_index += 1

                location.end_index += peek_index

                tokens.append(
                    Token(
                        token_type=TokenType.identifier,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case "=":
                tokens.append(
                    Token(
                        token_type=TokenType.equal_sign,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case ".":
                tokens.append(
                    Token(
                        token_type=TokenType.dot,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case ",":
                tokens.append(
                    Token(
                        token_type=TokenType.comma,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case "(":
                tokens.append(
                    Token(
                        token_type=TokenType.left_parenthesis,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case ")":
                tokens.append(
                    Token(
                        token_type=TokenType.right_parenthesis,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case "[":
                tokens.append(
                    Token(
                        token_type=TokenType.left_bracket,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case "]":
                tokens.append(
                    Token(
                        token_type=TokenType.right_bracket,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case "{":
                tokens.append(
                    Token(
                        token_type=TokenType.left_brace,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case "}":
                tokens.append(
                    Token(
                        token_type=TokenType.right_brace,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case " ":
                # peek all whitespace
                peek_index = 0
                while location.end_index + peek_index < len(source_code.content):
                    if source_code.content[location.end_index + peek_index] != " ":
                        break
                    peek_index += 1

                location.end_index += peek_index

                tokens.append(
                    Token(
                        token_type=TokenType.whitespace,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance()

            case "\n":
                tokens.append(
                    Token(
                        token_type=TokenType.new_line,
                        source_code=source_code,
                        source_code_location=deepcopy(location),
                    )
                )
                location.advance_line()

            case _:
                return InvalidSourceCodeCharacter(f"invalid character: {source_code.path}:{location.line_index}:{location.column_index}")

    return tokens


class NodeType(StrEnum):
    expr = auto()

    function = auto()
    apply = auto()

    lazy_record = auto()
    eager_record = auto()
    pair = auto()

    integer = auto()
    name = auto()


class AbstractSyntaxTreeNode:
    def __init__(self, node_type: NodeType, token: Token | None = None, nodes: list["AbstractSyntaxTreeNode"] | None = None) -> None:
        self._node_type = node_type
        self._token = token
        self._nodes = nodes or []

    @property
    def node_type(self) -> NodeType:
        return self._node_type

    @property
    def token(self) -> Token | None:
        return self._token

    @property
    def nodes(self) -> list["AbstractSyntaxTreeNode"]:
        return self._nodes

    def __str__(self) -> str:
        string = ""

        if self._token:
            string += f"{self.node_type} = {str(self._token)}"

        else:
            tmp_string = ""
            if self._nodes:
                for node in self._nodes:
                    tmp_string += f"{textwrap.indent(str(node), '  ')}\n"
            string += f"{self.node_type} = {{\n{tmp_string}}}"
        return string

    def pretty_string(self) -> str:
        string = ""

        if self.token:
            string += f"{str(self.token.literal)}"

        else:
            match self.node_type:
                case NodeType.expr:
                    string += f"({self.nodes[0].pretty_string()})"
                case NodeType.function:
                    string += f"{self.nodes[0].pretty_string()} . {self.nodes[1].pretty_string()}"
                case NodeType.apply:
                    string += f"{' '.join([node.pretty_string() for node in self.nodes])}"
                case NodeType.lazy_record:
                    string += f"{{{', '.join([node.pretty_string() for node in self.nodes])}}}"
                case NodeType.eager_record:
                    string += f"[{', '.join([node.pretty_string() for node in self.nodes])}]"
                case NodeType.pair:
                    string += f"{self.nodes[0].pretty_string()} = {self.nodes[1].pretty_string()}"
                case NodeType.integer:
                    string += f"{self.token.literal}"
                case NodeType.name:
                    string += f"{self.token.literal}"
        return string


def print_source_code_location(tokens: list[Token], index: int) -> str:
    if index < 0:
        index = 0
    elif index >= len(tokens):
        index = len(tokens) - 1

    return f"{tokens[index].source_code.path}:{tokens[index].source_code_location}"


class SyntaxParseError(Exception):
    def __init__(self, message: str, tokens: list[Token] | None = None, index: int | None = None, recoverable: bool = True, at_end=False) -> None:
        super().__init__(message)
        location = ""
        if tokens and index:
            self._message = f"{print_source_code_location(tokens, index)} - {message}\nTODO: snippet"
        else:
            self._message = f"{message}"
        self._recoverable = recoverable
        self._at_end = at_end

    @property
    def recoverable(self) -> bool:
        return self._recoverable

    @property
    def at_end(self) -> bool:
        return self._at_end

    def __str__(self) -> str:
        return ("recoverable" if self.recoverable else "unrecoverable") + f": {self._message}"


def peek_token(tokens: list[Token], index: int) -> Token | None:
    return tokens[index] if index < len(tokens) else None


def advance_token(tokens: list[Token], index: int) -> tuple[Token | SyntaxParseError, int]:
    token = peek_token(tokens, index)
    if token is None:
        error = SyntaxParseError(
            f"unexpected end of input",
            tokens=tokens,
            index=index,
            recoverable=False,
            at_end=True,
        )
        return error, index

    return token, index + 1


def match_token(tokens: list[Token], index: int, token_types: list[Token]) -> tuple[Token | None, int]:
    token = peek_token(tokens, index)

    if token is None:
        return None, index

    if token.token_type not in token_types:
        return None, index

    return token, index + 1


def expect_token(tokens: list[Token], index: int, token_types: list[Token]) -> tuple[Token | SyntaxParseError, int]:
    token = peek_token(tokens, index)

    if token is None:
        error = SyntaxParseError(
            f"expected {', '.join(f"'{token_type}'" for token_type in token_types)}, found <end of input>",
            tokens=tokens,
            index=index,
            recoverable=False,
            at_end=True,
        )
        return error, index

    if token.token_type not in token_types:
        error = SyntaxParseError(
            f"expected {', '.join(f"'{token_type}'" for token_type in token_types)} found {token.token_type}: {token.literal}",
            tokens=tokens,
            index=index,
        )
        return error, index

    return token, index + 1


def syntax_parse_name(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    name, index = expect_token(tokens, index, token_types=[TokenType.identifier])

    if type(name) == SyntaxParseError:
        return name, start_index

    return AbstractSyntaxTreeNode(NodeType.name, token=name), index


def syntax_parse_integer(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    integer, index = expect_token(tokens, index, token_types=[TokenType.integer])

    if type(integer) == SyntaxParseError:
        return integer, start_index

    return AbstractSyntaxTreeNode(NodeType.integer, token=integer), index


def syntax_parse_function(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode, int]:
    start_index = index

    first: Token | None = peek_token(tokens, index)
    second: Token | None = peek_token(tokens, index + 1)

    if not first or first.token_type != TokenType.identifier:
        error = SyntaxParseError(
            message=f"expected name in function: <name> . <expr>",
            tokens=tokens,
            index=index,
            recoverable=True
        )
        return error, start_index

    if not second or second.token_type != TokenType.dot:
        error = SyntaxParseError(
            message=f"expected dot in function: <name> . <expr>",
            tokens=tokens,
            index=index,
            recoverable=True,
        )
        return error, start_index

    name, index = advance_token(tokens, index)

    name = AbstractSyntaxTreeNode(NodeType.name, token=name)

    dot, index = advance_token(tokens, index)

    expr, index = syntax_parse_expr(tokens, index)

    if type(expr) is SyntaxParseError:
        if expr.recoverable:
            error = SyntaxParseError(
                message=f"expected expr in function: <name> . <expr>",
                tokens=tokens,
                index=index,
                recoverable=False,
            )
            return error, start_index
        else:
            return expr, start_index

    return AbstractSyntaxTreeNode(NodeType.function, nodes=[name, expr]), index


def syntax_parse_expr(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    function, index = syntax_parse_function(tokens, index)

    if type(function) is AbstractSyntaxTreeNode:
        return AbstractSyntaxTreeNode(NodeType.expr, nodes=[function]), index

    if type(function) is SyntaxParseError and not function.recoverable:
        return function, start_index

    apply, index = syntax_parse_apply(tokens, index)

    if type(apply) is SyntaxParseError:
        return apply, start_index

    return AbstractSyntaxTreeNode(NodeType.expr, nodes=[apply]), index


def syntax_parse_apply(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    basics: list[AbstractSyntaxTreeNode] = []

    first, index = syntax_parse_basic(tokens, index)

    if type(first) is SyntaxParseError:
        return first, start_index

    basics.append(first)

    while True:
        next_basic, index_2 = syntax_parse_basic(tokens, index)

        if type(next_basic) is SyntaxParseError:
            if not next_basic.recoverable:
                return next_basic, start_index
            else:
                break

        index = index_2

        basics.append(next_basic)

    return AbstractSyntaxTreeNode(NodeType.apply, nodes=basics), index


def syntax_parse_basic(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    token = peek_token(tokens, index)
    if token is None:
        error = SyntaxParseError(
            message=f"expected (int/name/(...)/{...}/[...]), found <end of input>",
            tokens=tokens,
            index=index,
            recoverable=True,
            at_end=True,
        )
        return error, start_index

    if token.token_type == TokenType.integer:
        return syntax_parse_integer(tokens, index)

    if token.token_type == TokenType.identifier:
        return syntax_parse_name(tokens, index)

    if token.token_type == TokenType.left_parenthesis:
        left_parenthesis, index = advance_token(tokens, index)

        next_token = peek_token(tokens, index)
        if not next_token:
            error = SyntaxParseError(
                f"expected apply or function, found <end of input>",
                tokens=tokens,
                index=index,
                recoverable=False,
            )
            return error, start_index

        if next_token.token_type == TokenType.right_parenthesis:
            error = SyntaxParseError(
                f"expected  apply or function, found {next_token.token_type}",
                tokens=tokens,
                index=index,
                recoverable=False,
            )
            return error, start_index

        expr, index = syntax_parse_expr(tokens, index)

        if type(expr) is SyntaxParseError:
            if expr.at_end and expr.recoverable:
                error = SyntaxParseError(
                    f"expected ')', found <end of input>",
                    tokens=tokens,
                    index=index,
                    recoverable=False,
                )
                return error, start_index
            return expr, start_index

        right_parenthesis, index = expect_token(tokens, index, token_types=[TokenType.right_parenthesis])

        if type(right_parenthesis) is SyntaxParseError:
            return right_parenthesis, start_index

        return expr, index

    if token.token_type == TokenType.left_brace:
        left_brace, index = advance_token(tokens, index)

        next_token = peek_token(tokens, index)

        if not next_token:
            error = SyntaxParseError(
                f"expected name or '}}', found <end of input>",
                tokens=tokens,
                index=index,
                recoverable=False
            )
            return error, start_index

        if next_token.token_type == TokenType.right_brace:
            next_token, index = advance_token(tokens, index)

            return AbstractSyntaxTreeNode(NodeType.lazy_record, nodes=[]), index

        if next_token.token_type != TokenType.identifier:
            error = SyntaxParseError(
                f"expected name, found {next_token.token_type}",
                tokens=tokens,
                index=index,
                recoverable=False,
            )
            return error, start_index

        pairs, index = syntax_parse_pairs(tokens, index)

        if type(pairs) is SyntaxParseError:
            return pairs, start_index

        right_brace, index = expect_token(tokens, index, token_types=[TokenType.right_brace])
        if type(right_brace) is SyntaxParseError:
            return right_brace, start_index

        return AbstractSyntaxTreeNode(NodeType.lazy_record, nodes=pairs), index

    if token.token_type == TokenType.left_bracket:
        left_bracket, index = advance_token(tokens, index)

        next_token = peek_token(tokens, index)

        if not next_token:
            error = SyntaxParseError(
                f"expected name or ']', found <end of input>",
                tokens=tokens,
                index=index,
                recoverable=False,
            )
            return error, start_index

        if next_token.token_type == TokenType.right_bracket:
            next_token, index = advance_token(tokens, index)

            return AbstractSyntaxTreeNode(NodeType.eager_record, nodes=[]), index

        if next_token.token_type != TokenType.identifier:
            error = SyntaxParseError(
                f"expected name, found {next_token.token_type}",
                tokens=tokens,
                index=index,
                recoverable=False,
            )
            return error, start_index

        pairs, index = syntax_parse_pairs(tokens, index)

        if type(pairs) is SyntaxParseError:
            return pairs, start_index

        right_bracket, index = expect_token(tokens, index, token_types=[TokenType.right_bracket])
        if type(right_bracket) is SyntaxParseError:
            return right_bracket, start_index

        return AbstractSyntaxTreeNode(NodeType.eager_record, nodes=pairs), index

    if token.token_type in [TokenType.dot, TokenType.equal_sign]:
        error = SyntaxParseError(
            f"expected (int/name/(...)/{{...}}/[...]), found {token.token_type}",
            tokens=tokens,
            index=index,
            recoverable=False,
        )
        return error, start_index

    error = SyntaxParseError(
        f"expected (int/name/(...)/{{...}}/[...]), found {token.token_type}",
        tokens=tokens,
        index=index,
    )
    return error, start_index


def syntax_parse_pairs(tokens: list[TokenType], index: int = 0) -> tuple[list[AbstractSyntaxTreeNode] | SyntaxParseError, int]:
    start_index = index

    pairs: list[AbstractSyntaxTreeNode] = []

    first, index = syntax_parse_pair(tokens, index)

    if type(first) is SyntaxParseError:
        return first, start_index

    pairs.append(first)

    while True:
        comma, index_2 = match_token(tokens, index, token_types=[TokenType.comma])
        if not comma:
            break

        next_pair, index = syntax_parse_pair(tokens, index_2)

        if type(next_pair) is SyntaxParseError:
            return next_pair, start_index

        pairs.append(next_pair)

    return pairs, index


def syntax_parse_pair(tokens: list[Token], index: int) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    name, index = syntax_parse_name(tokens, index)

    if type(name) is SyntaxParseError:
        return name, start_index

    equal_sign, index = expect_token(tokens, index, token_types=[TokenType.equal_sign])

    if type(equal_sign) is SyntaxParseError:
        return equal_sign, start_index

    value, index = syntax_parse_expr(tokens, index)

    if type(value) is SyntaxParseError:
        return value, start_index

    return AbstractSyntaxTreeNode(NodeType.pair, nodes=[name, value]), index


def syntax_parse_tokens(tokens: list[Token]) -> AbstractSyntaxTreeNode | SyntaxParseError:
    tokens_without_whitespace: list[AbstractSyntaxTreeNode] = []

    for token in tokens:
        match token.token_type:
            case w if w not in [TokenType.whitespace, TokenType.new_line]:
                tokens_without_whitespace.append(token)
            case _:
                continue

    if not tokens_without_whitespace:
        return SyntaxParseError("empty input", recoverable=False)

    # for token in tokens_without_whitespace:
    #     print(token)

    tokens = tokens_without_whitespace

    abstract_syntax_tree, index = syntax_parse_expr(tokens, 0)

    if type(abstract_syntax_tree) is SyntaxParseError:
        return abstract_syntax_tree

    # print(abstract_syntax_tree)

    if index != len(tokens):
        error = SyntaxParseError(
            f"unexpected '{tokens[index].token_type}'",
            tokens=tokens,
            index=index,
            recoverable=False,
        )
        return error

    return abstract_syntax_tree


class SemanticContext:
    def __init__(self, level: int = 0, bound_names: list[str] | None = None):
        self._bound_names = bound_names or []
        self._level = level

    @property
    def bound_names(self) -> list[str]:
        return self._bound_names

    @bound_names.setter
    def bound_names(self, bound_names: list[str]) -> None:
        self._bound_names = bound_names

    @property
    def level(self) -> int:
        return self._level

    def increase_level(self) -> None:
        self._level += 1


class SemanticNode:
    def __init__(self, node_type: NodeType, nodes: list['SemanticNode'] | None = None, name: str | None = None, integer: int | None = None, token: Token | None = None):
        self._node_type = node_type
        self._nodes = nodes
        self._name = name
        self._integer = integer
        self._token = token

    @property
    def node_type(self) -> NodeType:
        return self._node_type

    @property
    def nodes(self) -> list['SemanticNode']:
        return self._nodes

    @property
    def name(self) -> str:
        return self._name

    @property
    def integer(self) -> int:
        return self._integer

    @property
    def token(self) -> Token:
        return self._token

    def pretty_string(self) -> str:
        string = ""

        match self.node_type:
            case NodeType.expr:
                string += f"({self.nodes[0].pretty_string()})"
            case NodeType.function:
                string += f"{self.nodes[0].pretty_string()} . {self.nodes[1].pretty_string()}"
            case NodeType.apply:
                string += f"{' '.join([node.pretty_string() for node in self.nodes])}"
            case NodeType.lazy_record:
                string += f"{{{', '.join([node.pretty_string() for node in self.nodes])}}}"
            case NodeType.eager_record:
                string += f"[{', '.join([node.pretty_string() for node in self.nodes])}]"
            case NodeType.pair:
                string += f"{self.nodes[0].pretty_string()} = {self.nodes[1].pretty_string()}"
            case NodeType.integer:
                string += f"{self.integer}"
            case NodeType.name:
                string += f"{self.name}"
        return string


class SemanticParseError(Exception):
    def __init__(self, message):
        super().__init__(message)


def semantic_parse_expr(expr: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    node = expr.nodes[0]

    context.increase_level()

    match(node.node_type):
        case NodeType.function:
            parsed_node = semantic_parse_function(node, context=deepcopy(context))

        case NodeType.apply:
            parsed_node = semantic_parse_apply(node, context=deepcopy(context))

    if type(parsed_node) == SemanticParseError:
        return parsed_node

    return SemanticNode(NodeType.expr, nodes=[parsed_node])


def semantic_parse_function(function: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    name = function.nodes[0]
    expr = function.nodes[1]

    context.increase_level()

    parsed_name = semantic_parse_function_name(name, context=deepcopy(context))

    if type(parsed_name) == SemanticParseError:
        return parsed_name

    context.bound_names.append(parsed_name.name)

    parsed_expr = semantic_parse_expr(expr, context=deepcopy(context))

    if type(parsed_expr) == SemanticParseError:
        return parsed_expr

    return SemanticNode(NodeType.function, nodes=[parsed_name, parsed_expr])


def semantic_parse_apply(apply: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    basics = apply.nodes

    context.increase_level()

    parsed_basics: list[SemanticNode] = []

    for basic in basics:
        match(basic.node_type):
            case NodeType.integer:
                integer = semantic_parse_integer(basic, context=deepcopy(context))

                if type(integer) == SemanticParseError:
                    return integer

                parsed_basics.append(integer)

            case NodeType.name:
                name = semantic_parse_name(basic, context=deepcopy(context))

                if type(name) == SemanticParseError:
                    return name

                parsed_basics.append(name)

            case NodeType.expr:
                expr = semantic_parse_expr(basic, context=deepcopy(context))

                if type(expr) == SemanticParseError:
                    return expr

                parsed_basics.append(expr)

            case NodeType.lazy_record:
                lazy_record = semantic_parse_lazy_record(basic, context=deepcopy(context))

                if type(lazy_record) == SemanticParseError:
                    return lazy_record

                context.bound_names.extend([pair.nodes[0].name for pair in lazy_record.nodes])
                parsed_basics.append(lazy_record)

            case NodeType.eager_record:
                eager_record = semantic_parse_eager_record(basic, context=deepcopy(context))

                if type(eager_record) == SemanticParseError:
                    return eager_record

                context.bound_names.extend([pair.nodes[0].name for pair in eager_record.nodes])
                parsed_basics.append(eager_record)

    return SemanticNode(NodeType.apply, nodes=parsed_basics)


def semantic_parse_lazy_record(lazy_record: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    pairs = lazy_record.nodes

    context.increase_level()

    parsed_pairs: list[SemanticNode] = []

    for pair in pairs:
        parsed_pair = semantic_parse_pair(pair, context=deepcopy(context))

        if type(parsed_pair) == SemanticParseError:
            return parsed_pair

        context.bound_names.append(parsed_pair.nodes[0].name)
        parsed_pairs.append(parsed_pair)

    return SemanticNode(NodeType.lazy_record, nodes=parsed_pairs)


def semantic_parse_eager_record(eager_record: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    pairs = eager_record.nodes

    context.increase_level()

    parsed_pairs: list[SemanticNode] = []

    for pair in pairs:
        parsed_pair = semantic_parse_pair(pair, context=deepcopy(context))

        if type(parsed_pair) == SemanticParseError:
            return parsed_pair

        context.bound_names.append(parsed_pair.nodes[0].name)
        parsed_pairs.append(parsed_pair)

    return SemanticNode(NodeType.eager_record, nodes=parsed_pairs)


def semantic_parse_pair(pair: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    name = pair.nodes[0]
    expr = pair.nodes[1]

    context.increase_level()

    parsed_name = semantic_parse_pair_name(name, context=deepcopy(context))
    if type(parsed_name) == SemanticParseError:
        return parsed_name

    if parsed_name.name in ["plus", "minus", "mult", "div", "cond"]:
        return SemanticParseError(f"{parsed_name.name} cannot be a binding name in a record", parsed_name)

    context.bound_names.append(parsed_name.name)

    parsed_expr = semantic_parse_expr(expr, context=deepcopy(context))
    if type(parsed_expr) == SemanticParseError:
        return parsed_expr

    return SemanticNode(NodeType.pair, nodes=[parsed_name, parsed_expr])


def semantic_parse_integer(integer: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    context.increase_level()

    return SemanticNode(NodeType.integer, integer=int(integer.token.literal), token=integer.token)


def semantic_parse_pair_name(name: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    context.increase_level()

    return SemanticNode(NodeType.name, name=name.token.literal, token=name.token)


def semantic_parse_function_name(name: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    context.increase_level()

    return SemanticNode(NodeType.name, name=name.token.literal, token=name.token)


def semantic_parse_name(name: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    context.increase_level()

    # forbid free names
    # if name.token.literal not in context.bound_names:
    #     return SemanticParseError(f"name '{name.token.literal}' not bound in scope {context.bound_names}")

    return SemanticNode(NodeType.name, name=name.token.literal, token=name.token)


class EvalContext:
    def __init__(self, level: int = 0):
        self._level = level

    @property
    def level(self) -> int:
        return self._level

    def increase_level(self) -> None:
        self._level += 1


class EvalError(Exception):
    def __init__(self, message):
        super().__init__(message)


Environment = dict[str, SemanticNode]


def eval_expr(expr: SemanticNode, environment: Environment | None = None, context: EvalContext = EvalContext()) -> SemanticNode | EvalError:
    # print(f"{' ' * 2 * context.level}eval expr ", expr.pretty_string())
    context.increase_level()

    environment = {} if environment is None else environment

    node = expr.nodes[0]

    match(node.node_type):
        case NodeType.function:
            evaluated_node = eval_function(node, copy(environment), context=deepcopy(context))
        case NodeType.apply:
            evaluated_node = eval_apply(node, copy(environment), context=deepcopy(context))

    if type(evaluated_node) == EvalError:
        return evaluated_node

    return SemanticNode(NodeType.expr, nodes=[evaluated_node])


def eval_function(function: SemanticNode, environment: Environment, context: EvalContext) -> SemanticNode | EvalError:
    # print(f"{' ' * 2 * context.level}eval function: ", function.pretty_string())
    context.increase_level()

    name = function.nodes[0]
    expr = function.nodes[1]

    # shadow name
    environment.pop(name.name, None)

    evaluated_expr = eval_expr(expr, copy(environment), context=deepcopy(context))

    if type(evaluated_expr) == EvalError:
        return evaluated_expr

    return SemanticNode(NodeType.function, nodes=[name, evaluated_expr])


def eval_apply(apply: SemanticNode, environment: Environment, context: EvalContext) -> SemanticNode | EvalError:
    # print(f"{' ' * 2 * context.level}eval apply: ", apply.pretty_string())
    context.increase_level()

    basics = apply.nodes

    evaluated_basics = eval_basics(basics, copy(environment), context=deepcopy(context))

    if type(evaluated_basics) == EvalError:
        return evaluated_basics

    return SemanticNode(NodeType.apply, nodes=evaluated_basics)


def eval_basics(basics: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    # print(f"{' ' * 2 * context.level}eval basics: ", [basic.pretty_string() for basic in basics])
    context.increase_level()

    evaluated_basic: SemanticNode = eval_basic(basics[0], copy(environment), context=deepcopy(context))

    if type(evaluated_basic) == EvalError:
        return evaluated_basic

    match(evaluated_basic.node_type):
        case NodeType.name:
            match(evaluated_basic.name):
                case x if x in ["plus", "minus", "mult", "div"]:
                    if len(basics[1:]) != 2:
                        return EvalError(f"{evaluated_basic.name} needs to have exactly 2 arguments")
                    else:
                        pass
                case "cond":
                    if len(basics[1:]) != 3:
                        return EvalError(f"{evaluated_basic.name} needs to have exactly 3 arguments")
                    else:
                        pass
                case _:
                    pass

        case x if x in [NodeType.expr, NodeType.lazy_record, NodeType.eager_record]:
            pass

    if len(basics) == 1:
        return [evaluated_basic]

    return apply_arguments_on_basic(evaluated_basic, basics[1:], copy(environment), context=deepcopy(context))


def eval_basic(basic: SemanticNode, environment: Environment, context: EvalContext) -> SemanticNode | EvalError:
    # print(f"{' ' * 2 * context.level}eval basic: ", basic.pretty_string())
    context.increase_level()

    match(basic.node_type):
        case NodeType.integer:
            return basic

        case NodeType.name:
            expr = environment.get(basic.name)

            if expr is None:
                return basic

            evaluated_basic = eval_basic(expr, copy(environment), deepcopy(context))

            if type(evaluated_basic) == EvalError:
                return evaluated_basic

            return evaluated_basic

        case NodeType.expr:
            evaluated_expr = eval_expr(basic, copy(environment), context=deepcopy(context))

            if type(evaluated_expr) == EvalError:
                return evaluated_expr

            return reduce_expr(evaluated_expr, context=deepcopy(context))

        case NodeType.lazy_record:
            return basic

        case NodeType.eager_record:
            return eval_eager_record(basic, copy(environment), context=deepcopy(context))


def reduce_expr(expr: SemanticNode, context: EvalContext) -> SemanticNode:
    node = expr.nodes[0]

    if node.node_type != NodeType.apply:
        return expr

    apply = node

    if len(apply.nodes) != 1:
        return expr

    return apply.nodes[0]


def eval_eager_record(eager_record: SemanticNode, environment: Environment, context: EvalContext) -> SemanticNode | EvalError:
    # print(f"{' ' * 2 * context.level}eval eager-record: ", eager_record.pretty_string())
    context.increase_level()

    pairs = eager_record.nodes

    evaluated_pairs = eval_pairs(pairs, copy(environment), context=deepcopy(context))

    if type(evaluated_pairs) == EvalError:
        return evaluated_pairs

    return SemanticNode(NodeType.lazy_record, nodes=evaluated_pairs)


def eval_pairs(pairs: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    if len(pairs) == 0:
        return []

    evaluated_pairs = []

    for pair in pairs:
        evaluated_pair = eval_pair(pair, copy(environment), context=deepcopy(context))

        if type(evaluated_pair) == EvalError:
            return evaluated_pair

        name = evaluated_pair.nodes[0]
        expr = evaluated_pair.nodes[1]

        environment[name.name] = expr

        # print(json.dumps({k: v.pretty_string() for k, v in environment.items()}, sort_keys=True, indent=4))

        evaluated_pairs.append(evaluated_pair)

    return evaluated_pairs


def eval_pair(pair: SemanticNode, environment: Environment, context: EvalContext) -> SemanticNode | EvalError:
    # print(f"{' ' * 2 * context.level}eval pair: ", pair.pretty_string())
    context.increase_level()

    name = pair.nodes[0]
    expr = pair.nodes[1]

    evaluated_expr = eval_expr(expr, copy(environment), context=deepcopy(context))

    if type(evaluated_expr) == EvalError:
        return evaluated_expr

    reduced_expr = reduce_expr(evaluated_expr, context=deepcopy(context))

    return SemanticNode(NodeType.pair, nodes=[name, reduced_expr])


def apply_arguments_on_basic(basic: SemanticNode, arguments: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    # print(f"{' ' * 2 * context.level}apply basic:  ", basic.pretty_string())
    # print(f"{' ' * 2 * context.level}on arguments: ", [argument.pretty_string() for argument in arguments])
    context.increase_level()

    match(basic.node_type):
        case NodeType.integer:
            if len(arguments) != 0:
                return EvalError("integer cannot be applied on arguments")
            else:
                return basic

        case NodeType.name:
            match(basic.name):
                case x if x in ["plus", "minus", "mult", "div"]:
                    return apply_arguments_on_arithmetic_function(x, arguments, copy(environment), context=deepcopy(context))
                case "cond":
                    return apply_arguments_on_cond(arguments, copy(environment), context=deepcopy(context))
                case _:
                    return [basic] + arguments

        case NodeType.expr:
            return apply_arguments_on_expr(basic, arguments, copy(environment), context=deepcopy(context))

        case x if x in [NodeType.lazy_record, NodeType.eager_record]:
            return apply_arguments_on_record(basic, arguments, copy(environment), context=deepcopy(context))


def apply_arguments_on_arithmetic_function(operation_name: str, arguments: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    # print(f"{' ' * 2 * context.level}apply arithmetic: ", operation_name)
    # print(f"{' ' * 2 * context.level}on arguments:     ", [argument.pretty_string() for argument in arguments])
    context.increase_level()

    if len(arguments) != 2:
        return EvalError(f"{operation_name} needs to have exactly 2 arguments")

    argument_1 = eval_basic(arguments[0], copy(environment), context=deepcopy(context))

    if type(argument_1) == EvalError:
        return argument_1

    argument_2 = eval_basic(arguments[1], copy(environment), context=deepcopy(context))

    if type(argument_2) == EvalError:
        return argument_2

    if argument_1.node_type != NodeType.integer:
        return [SemanticNode(node_type=NodeType.name, name=operation_name), argument_1, argument_2]

    if argument_2.node_type != NodeType.integer:
        return [SemanticNode(node_type=NodeType.name, name=operation_name), argument_1, argument_2]

    match(operation_name):
        case "plus":
            return [SemanticNode(NodeType.integer, integer=argument_1.integer + argument_2.integer)]
        case "minus":
            return [SemanticNode(NodeType.integer, integer=argument_1.integer - argument_2.integer)]
        case "mult":
            return [SemanticNode(NodeType.integer, integer=argument_1.integer * argument_2.integer)]
        case "div":
            return [SemanticNode(NodeType.integer, integer=argument_1.integer // argument_2.integer)]


def apply_arguments_on_cond(arguments: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    # print(f"{' ' * 2 * context.level}apply basic:  ", "cond")
    # print(f"{' ' * 2 * context.level}on arguments: ", [argument.pretty_string() for argument in arguments])
    context.increase_level()

    if len(arguments) != 3:
        return EvalError(f"cond needs to have exactly 3 arguments")

    argument_1 = eval_basic(arguments[0], copy(environment), context=deepcopy(context))

    if type(argument_1) == EvalError:
        return argument_1

    if is_equivalent_to_true(argument_1):
        result = eval_basic(arguments[1], copy(environment), context=deepcopy(context))
    elif is_equivalent_to_false(argument_1):
        result = eval_basic(arguments[2], copy(environment), context=deepcopy(context))
    else:
        return [SemanticNode(node_type=NodeType.name, name="cond"), argument_1] + arguments[1:]

    if type(result) == EvalError:
        return result

    return [result]


def is_equivalent_to_true(basic: SemanticNode) -> bool:
    match(basic.node_type):
        case NodeType.integer:
            return basic.integer != 0
        case x if x in [NodeType.lazy_record, NodeType.eager_record]:
            if len(basic.nodes) == 0:
                return False
            else:
                return True
        case _:
            return False


def is_equivalent_to_false(basic: SemanticNode) -> bool:
    match(basic.node_type):
        case NodeType.integer:
            return basic.integer == 0
        case x if x in [NodeType.lazy_record, NodeType.eager_record]:
            if len(basic.nodes) == 0:
                return True
            else:
                return False
        case _:
            return False


def apply_arguments_on_expr(expr: SemanticNode, arguments: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    # print(f"{' ' * 2 * context.level}apply expr:   ", expr.pretty_string())
    # print(f"{' ' * 2 * context.level}on arguments: ", [argument.pretty_string() for argument in arguments])
    context.increase_level()

    node = expr.nodes[0]

    match(node.node_type):
        case NodeType.function:
            return apply_arguments_on_function(node, arguments, copy(environment), context=deepcopy(context))
        case NodeType.apply:
            return [expr] + arguments


def apply_arguments_on_function(function: SemanticNode, arguments: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    # print(f"{' ' * 2 * context.level}apply function: ", function.pretty_string())
    # print(f"{' ' * 2 * context.level}on arguments:   ", [argument.pretty_string() for argument in arguments])
    context.increase_level()

    name = function.nodes[0]
    expr = function.nodes[1]

    argument = arguments[0]

    argument = eval_basic(argument, copy(environment), context=deepcopy(context))

    if type(argument) == EvalError:
        return argument

    expr = replace_name_with_basic_in_expr(name.name, argument, expr)

    evaluated_expr = eval_expr(expr, copy(environment), context=deepcopy(context))

    if type(evaluated_expr) == EvalError:
        return evaluated_expr

    reduced_expr = reduce_expr(evaluated_expr, context=deepcopy(context))

    return eval_basics([reduced_expr] + arguments[1:], copy(environment), context=deepcopy(context))


def apply_arguments_on_record(record: SemanticNode, arguments: list[SemanticNode], environment: Environment, context: EvalContext) -> list[SemanticNode] | EvalError:
    # print(f"{' ' * 2 * context.level}apply record: ", record.pretty_string())
    # print(f"{' ' * 2 * context.level}on arguments: ", [argument.pretty_string() for argument in arguments])
    context.increase_level()

    for pair in reversed(record.nodes):
        name = pair.nodes[0]
        expr = pair.nodes[1]

        environment[name.name] = expr

    # print(json.dumps({k: v.pretty_string() for k, v in environment.items()}, sort_keys=True, indent=4))

    return eval_basics(arguments, copy(environment), context=deepcopy(context))


def replace_name_with_basic_in_expr(name: str, argument: SemanticNode, expr: SemanticNode) -> SemanticNode:
    node = expr.nodes[0]

    match(node.node_type):
        case NodeType.function:
            node = replace_name_with_basic_in_function(name, argument, node)
        case NodeType.apply:
            node = replace_name_with_basic_in_apply(name, argument, node)

    return SemanticNode(NodeType.expr, [node])


def replace_name_with_basic_in_function(name: str, argument: SemanticNode, function: SemanticNode) -> SemanticNode:
    function_name = function.nodes[0]
    function_expr = function.nodes[1]

    if name == function_name.name:
        return function

    function_expr = replace_name_with_basic_in_expr(name, argument, function_expr)

    return SemanticNode(NodeType.function, [function_name, function_expr])


def replace_name_with_basic_in_apply(name: str, argument: SemanticNode, apply: SemanticNode) -> SemanticNode:
    basics = apply.nodes
    basics = replace_name_with_basic_in_basics(name, argument, basics)

    return SemanticNode(NodeType.apply, basics)


def replace_name_with_basic_in_basics(name: str, argument: SemanticNode, basics: list[SemanticNode]) -> list[SemanticNode]:
    replaced_basics = []

    for basic in basics:
        replaced_basic = replace_name_with_basic_in_basic(name, argument, basic)
        replaced_basics.append(replaced_basic)

    return replaced_basics


def replace_name_with_basic_in_basic(name: str, argument: SemanticNode, basic: SemanticNode) -> SemanticNode:
    match(basic.node_type):
        case NodeType.integer:
            return basic
        case NodeType.name:
            if basic.name == name:
                return argument
            else:
                return basic
        case NodeType.expr:
            replaced_expr = replace_name_with_basic_in_expr(name, argument, basic)
            return replaced_expr
        case NodeType.lazy_record:
            replaced_record = replace_name_with_basic_in_lazy_record(name, argument, basic)
            return replaced_record
        case NodeType.eager_record:
            replaced_record = replace_name_with_basic_in_eager_record(name, argument, basic)
            return replaced_record


def replace_name_with_basic_in_lazy_record(name: str, argument: SemanticNode, lazy_record: SemanticNode) -> SemanticNode:
    pairs = lazy_record.nodes
    pairs = replace_name_with_basic_in_pairs(name, argument, pairs)

    return SemanticNode(NodeType.lazy_record, pairs)


def replace_name_with_basic_in_eager_record(name: str, argument: SemanticNode, eager_record: SemanticNode) -> SemanticNode:
    pairs = eager_record.nodes
    pairs = replace_name_with_basic_in_pairs(name, argument, pairs)

    return SemanticNode(NodeType.eager_record, pairs)


def replace_name_with_basic_in_pairs(name: str, argument: SemanticNode, pairs: list[SemanticNode]) -> list[SemanticNode]:
    replaced_pairs = []

    for pair in pairs:
        replaced_pair = replace_name_with_basic_in_pair(name, argument, pair)
        replaced_pairs.append(replaced_pair)

    return replaced_pairs


def replace_name_with_basic_in_pair(name: str, argument: SemanticNode, pair: SemanticNode) -> SemanticNode:
    pair_name = pair.nodes[0]
    pair_expr = pair.nodes[1]

    pair_expr = replace_name_with_basic_in_expr(name, argument, pair_expr)

    return SemanticNode(NodeType.pair, nodes=[pair_name, pair_expr])


def main() -> None:
    parser = argparse.ArgumentParser(description="f1 intperpreter")
    parser.add_argument("filename", help="Path to the file to interpret")
    args = parser.parse_args()

    source_code = SourceCode(
        path=args.filename,
        content=read_source_code(args.filename),
    )

    tokens = group_characters(source_code)
    if type(tokens) is InvalidSourceCodeCharacter:
        print(tokens, file=sys.stderr)
        exit(1)

    abstract_syntax_tree = syntax_parse_tokens(tokens)

    if type(abstract_syntax_tree) is SyntaxParseError:
        print(abstract_syntax_tree, file=sys.stderr)
        exit(1)

    semantic_context = SemanticContext(bound_names=['plus', 'minus', 'mult', 'div', 'cond'])

    semantic_tree = semantic_parse_expr(abstract_syntax_tree, semantic_context)

    if type(semantic_tree) is SemanticParseError:
        print(semantic_tree, file=sys.stderr)
        exit(1)

    # print(semantic_tree.pretty_string())

    evaluated_expr = eval_expr(semantic_tree)

    if type(evaluated_expr) is EvalError:
        print(evaluated_expr, file=sys.stderr)
        exit(1)

    print(evaluated_expr.pretty_string())


if __name__ == "__main__":
    sys.setrecursionlimit(10_000_000)

    main()
