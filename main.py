#!/usr/bin/python3.12
import argparse
import sys
from pathlib import Path
from enum import StrEnum, auto
from copy import deepcopy
import textwrap


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

                token_type = TokenType.identifier

                tokens.append(
                    Token(
                        token_type=token_type,
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


class AbstractSyntaxTreeNodeType(StrEnum):
    expr = auto()

    function = auto()
    apply = auto()

    lazy_record = auto()
    eager_record = auto()
    pair = auto()

    integer = auto()
    name = auto()


class AbstractSyntaxTreeNode:
    def __init__(self, node_type: AbstractSyntaxTreeNodeType, nodes: list["AbstractSyntaxTreeNode"] | None = None, token: Token | None = None) -> None:
        self._node_type = node_type
        self._token = token
        self.nodes = nodes or []

    @property
    def node_type(self) -> AbstractSyntaxTreeNodeType:
        return self._node_type

    @property
    def token(self) -> Token | None:
        return self._token

    @property
    def nodes(self) -> list["AbstractSyntaxTreeNode"]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: list["AbstractSyntaxTreeNode"]) -> None:
        self._nodes = nodes

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
                case AbstractSyntaxTreeNodeType.expr:
                    string += f"({self.nodes[0].pretty_string()})"
                case AbstractSyntaxTreeNodeType.function:
                    string += f"{self.nodes[0].pretty_string()} . {self.nodes[1].pretty_string()}"
                case AbstractSyntaxTreeNodeType.apply:
                    string += f"{' '.join([node.pretty_string() for node in self.nodes])}"
                case AbstractSyntaxTreeNodeType.lazy_record:
                    string += f"{{{', '.join([node.pretty_string() for node in self.nodes])}}}"
                case AbstractSyntaxTreeNodeType.eager_record:
                    string += f"[{', '.join([node.pretty_string() for node in self.nodes])}]"
                case AbstractSyntaxTreeNodeType.pair:
                    string += f"{self.nodes[0].pretty_string()} = {self.nodes[1].pretty_string()}"
                case AbstractSyntaxTreeNodeType.integer:
                    string += f"{self.token.literal}"
                case AbstractSyntaxTreeNodeType.name:
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
        return SyntaxParseError(
            f"expected {', '.join(f"'{token_type}'" for token_type in token_types)}, found <end of input>",
            tokens=tokens,
            index=index,
            recoverable=False,
            at_end=True,
        ), index

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

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.name, token=name), index


def syntax_parse_integer(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    integer, index = expect_token(tokens, index, token_types=[TokenType.integer])

    if type(integer) == SyntaxParseError:
        return integer, start_index

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.integer, token=integer), index


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

    name = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.name, token=name)

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

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.function, nodes=[name, expr]), index


def syntax_parse_expr(tokens: list[Token], index: int = 0) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
    start_index = index

    function, index = syntax_parse_function(tokens, index)

    if type(function) is AbstractSyntaxTreeNode:
        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.expr, nodes=[function]), index

    if type(function) is SyntaxParseError and not function.recoverable:
        return function, start_index

    apply, index = syntax_parse_apply(tokens, index)

    if type(apply) is SyntaxParseError:
        return apply, start_index

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.expr, nodes=[apply]), index


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

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.apply, nodes=basics), index


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
        integer, index = syntax_parse_integer(tokens, index)

        return integer, index

    if token.token_type == TokenType.identifier:
        name, index = syntax_parse_name(tokens, index)

        return name, index

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

            return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.lazy_record, nodes=[]), index

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

        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.lazy_record, nodes=pairs), index

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

            return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.eager_record, tokens=[]), index

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

        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.lazy_record, nodes=pairs), index

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

    def syntax_parse_one_pair(tokens: list[Token], index: int) -> tuple[AbstractSyntaxTreeNode | SyntaxParseError, int]:
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

        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.pair, nodes=[name, value]), index

    first, index = syntax_parse_one_pair(tokens, index)

    if type(first) is SyntaxParseError:
        return first, start_index

    pairs.append(first)

    while True:
        comma, index_2 = match_token(tokens, index, token_types=[TokenType.comma])
        if not comma:
            break

        next_pair, index = syntax_parse_one_pair(tokens, index_2)

        if type(next_pair) is SyntaxParseError:
            return next_pair, start_index

        pairs.append(next_pair)

    return pairs, index


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

    for token in tokens_without_whitespace:
        print(token)

    tokens = tokens_without_whitespace

    abstract_syntax_tree, index = syntax_parse_expr(tokens, 0)

    if type(abstract_syntax_tree) is SyntaxParseError:
        return abstract_syntax_tree

    print(abstract_syntax_tree)

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

    @level.setter
    def level(self, level: int) -> None:
        self._level = level


class SemanticNodeType(StrEnum):
    expr = auto()

    function = auto()
    apply = auto()

    lazy_record = auto()
    eager_record = auto()
    pair = auto()

    integer = auto()
    name = auto()


class SemanticNode:
    def __init__(self, node_type: SemanticNodeType, nodes: list['SemanticNode'] | None = None, name: str | None = None, integer: int | None = None):
        self._node_type = node_type
        self._nodes = nodes
        self._name = name
        self._integer = integer

    @property
    def node_type(self) -> SemanticNodeType:
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


class SemanticParseError(Exception):
    def __init__(self, message):
        super().__init__(message)


def semantic_parse_expr(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.expr

    nodes: list[SemanticNode] = []

    node = ast.nodes[0]

    context.level += 1

    match(node.node_type):
        case AbstractSyntaxTreeNodeType.function:
            function = semantic_parse_function(node, context=deepcopy(context))
            if type(function) == SemanticParseError:
                return function
            nodes.append(function)
        case AbstractSyntaxTreeNodeType.apply:
            apply = semantic_parse_apply(node, context=deepcopy(context))
            if type(apply) == SemanticParseError:
                return apply
            nodes.append(apply)

    return SemanticNode(SemanticNodeType.expr, nodes=nodes)


def semantic_parse_function(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.function

    nodes: list[SemanticNode] = []

    context.level += 1

    name = semantic_parse_function_name(ast.nodes[0], context=deepcopy(context))
    if type(name) == SemanticParseError:
        return name
    nodes.append(name)

    context.bound_names.append(name.name)

    expr = semantic_parse_expr(ast.nodes[1], context=deepcopy(context))
    if type(expr) == SemanticParseError:
        return expr
    nodes.append(expr)

    return SemanticNode(SemanticNodeType.function, nodes=[name, expr])


def semantic_parse_apply(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.apply

    nodes: list[SemanticNode] = []

    context.level += 1

    for node in ast.nodes:
        match(node.node_type):
            case AbstractSyntaxTreeNodeType.integer:
                integer = semantic_parse_integer(node, context=deepcopy(context))
                if type(integer) == SemanticParseError:
                    return integer
                nodes.append(integer)
            case AbstractSyntaxTreeNodeType.name:
                name = semantic_parse_name(node, context=deepcopy(context))
                if type(name) == SemanticParseError:
                    return name
                nodes.append(name)
            case AbstractSyntaxTreeNodeType.expr:
                expr = semantic_parse_expr(node, context=deepcopy(context))
                if type(expr) == SemanticParseError:
                    return expr
                nodes.append(expr)
            case AbstractSyntaxTreeNodeType.lazy_record:
                eager_record = semantic_parse_lazy_record(node, context=deepcopy(context))
                if type(eager_record) == SemanticParseError:
                    return eager_record

                context.bound_names.extend([pairs.nodes[0].token.literal for pairs in node.nodes])
                nodes.append(eager_record)
            case AbstractSyntaxTreeNodeType.eager_record:
                eager_record = semantic_parse_eager_record(node, context=deepcopy(context))
                if type(eager_record) == SemanticParseError:
                    return eager_record

                context.bound_names.extend([pairs.nodes[0].literal for pairs in node.nodes])
                nodes.append(eager_record)

    return SemanticNode(SemanticNodeType.apply, nodes=nodes)


def semantic_parse_lazy_record(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.lazy_record

    nodes: list[SemanticNode] = []

    context.level += 1

    for node in ast.nodes:
        pair = semantic_parse_pair(node, context=deepcopy(context))
        if type(pair) == SemanticParseError:
            return pair
        nodes.append(pair)

        context.bound_names.append(node.nodes[0].token.literal)

    return SemanticNode(SemanticNodeType.lazy_record, nodes=nodes)


def semantic_parse_eager_record(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.eager_record

    nodes: list[SemanticNode] = []

    context.level += 1

    for node in ast.nodes:
        pair = semantic_parse_pair(node, context=deepcopy(context))
        if type(pair) == SemanticParseError:
            return pair
        nodes.append(pair)

        context.bound_names.append(ast.nodes[0].literal)

    return SemanticNode(SemanticNodeType.eager_record, nodes=nodes)


def semantic_parse_pair(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.pair

    nodes: list[SemanticNode] = []

    context.level += 1

    name = semantic_parse_pair_name(ast.nodes[0], context=deepcopy(context))
    if type(name) == SemanticParseError:
        return name

    nodes.append(name)

    expr = semantic_parse_expr(ast.nodes[1], context=deepcopy(context))
    if type(expr) == SemanticParseError:
        return expr

    nodes.append(expr)

    return SemanticNode(SemanticNodeType.pair, nodes=nodes)


def semantic_parse_integer(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.integer

    context.level += 1

    return SemanticNode(SemanticNodeType.integer, integer=int(ast.token.literal))


def semantic_parse_pair_name(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.name

    context.level += 1

    return SemanticNode(SemanticNodeType.name, name=ast.token.literal)


def semantic_parse_function_name(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.name

    context.level += 1

    return SemanticNode(SemanticNodeType.name, name=ast.token.literal)


def semantic_parse_name(ast: AbstractSyntaxTreeNode, context: SemanticContext) -> SemanticNode | SemanticParseError:
    assert type(ast) == AbstractSyntaxTreeNode
    assert ast.node_type == AbstractSyntaxTreeNodeType.name

    context.level += 1

    if ast.token.literal not in context.bound_names:
        return SemanticParseError("name not bound")

    return SemanticNode(SemanticNodeType.name, name=ast.token.literal)


class EvalError(Exception):
    def __init__(self, message):
        super().__init__(message)


def eval_expr(expr: SemanticNode) -> SemanticNode | EvalError:
    node = expr.nodes[0]

    match(node.node_type):
        case SemanticNodeType.function:
            evaluated_node = eval_function(node)
        case SemanticNodeType.apply:
            evaluated_node = eval_apply(node)

    if type(evaluated_node) == EvalError:
        return evaluated_node

    return SemanticNode(SemanticNodeType.expr, nodes=[evaluated_node])


def eval_function(function: SemanticNode) -> SemanticNode | EvalError:
    name = function.nodes[0]
    expr = function.nodes[1]

    evaluated_expr = eval_expr(expr)

    if type(evaluated_expr) == EvalError:
        return evaluated_expr

    return SemanticNode(SemanticNodeType.function, nodes=[name, evaluated_expr])


def eval_apply(apply: SemanticNode) -> SemanticNode | EvalError:
    basics = apply.nodes

    evaluated_basics = eval_basics(basics)

    if type(evaluated_basic) == EvalError:
        return evaluated_basic

    return SemanticNode(SemanticNodeType.apply, nodes=evaluated_pairs)


def eval_basics(basics: list[SemanticNode]) -> list[SemanticNode] | EvalError:
    evaluated_basic = eval_basic(basics[0])

    if type(evaluated_basic) == EvalError:
        return evaluated_basic

    if len(basics) == 1:
        return [evaluated_basic]

    reduced_basics = apply_arguments_on_function(evaluated_basic, basics[1:])

    if type(reduced_basics) == EvalError:
        return reduced_basics

    return eval_basics(reduced_basics)


def eval_eager_record(eager_record: SemanticNode) -> SemanticNode | EvalError:
    pairs = eager_record.nodes

    evaluated_pairs = eval_pairs(pairs)

    if type(evaluated_pairs) == EvalError:
        return evaluated_pairs

    return SemanticNode(SemanticNodeType.lazy_record, nodes=evaluated_pairs)


def eval_pairs(pairs: list[SemanticNode]) -> list[SemanticNode] | EvalError:
    evaluated_pair = eval_pair(pairs[0])

    if type(evaluated_pair) == EvalError:
        return evaluated_pair

    if len(pairs) == 1:
        return [evaluated_pair]

    name = evaluated_pair.nodes[0]
    expr = evaluated_pair.nodes[1]

    replaced_pairs = replace_name_with_expr_in_pairs(name, expr, pairs[1:])

    evaluated_pairs = eval_pairs(replaced_pairs)

    if type(evaluated_pairs) == EvalError:
        return evaluated_pairs

    return [evaluated_pair] + evaluated_pairs


def eval_pair(pair: SemanticNode) -> SemanticNode | EvalError:
    name = pair.nodes[0]
    expr = pair.nodes[1]

    evaluated_expr = eval_expr(expr)

    if type(evaluated_expr) == EvalError:
        return evaluated_expr

    return SemanticNode(SemanticNodeType.pair, nodes=[name, evaluated_expr])


def apply_arguments_on_basic(basic: SemanticNode, arguments: list[SemanticNode]) -> list[SemanticNode] | EvalError:
    match(basic.node_type):
        case SemanticNodeType.integer:
            if len(arguments) != 0:
                return EvalError("integer cannot be applied on arguments")
            else:
                return basic
        case SemanticNodeType.name:
            match(basic.name):
                case "plus":
                    pass
                case "minus":
                    pass
                case "mult":
                    pass
                case "div":
                    pass
                case "cond":
                    pass
        case SemanticNodeType.expr:
            pass
        case SemanticNodeType.lazy_record:
            pass
        case SemanticNodeType.eager_record:
            pass


def replace_name_with_expr_in_pairs(name: SemanticNode, expr: SemanticNode, pairs: list[SemanticNode]) -> list[SemanticNode] | EvalError:
    replaced_pairs = []

    for pair in pairs:
        replaced_pair = replace_name_with_expr_in_pair(name, expr, pair)

        if type(replaced_pair) == EvalError:
            return replaced_pair

        replaced_pairs.append(replaced_pair)

    return replaced_pairs


def replace_name_with_expr_in_pair(name: SemanticNode, expr: SemanticNode, pair: SemanticNode) -> SemanticNode | EvalError:
    pair_name = pair.nodes[0]
    pair_expr = pair.nodes[1]

    replaced_pair_expr = replace_name_with_expr_in_expr(name, expr, pair_expr)

    if type(replaced_pair_expr) == EvalError:
        return replaced_pair_expr

    return SemanticNode(SemanticNodeType.expr, nodes=[pair_name, replaced_pair_expr])


def replace_name_with_expr_in_expr(name: SemanticNode, expr: SemanticNode, expr1: SemanticNode) -> SemanticNode | EvalError:
    node = expr1.nodes[0]

    match(node.node_type):
        case SemanticNodeType.function:
            replaced_node = replace_name_with_expr_in_function(name, expr, node)
        case SemanticNodeType.apply:
            replaced_node = replace_name_with_expr_in_apply(name, expr, node)

    if type(replaced_node) == EvalError:
        return replaced_node

    return SemanticNode(SemanticNodeType.expr, nodes=[replaced_node])


def replace_name_with_expr_in_function(name: SemanticNode, expr: SemanticNode, function: SemanticNode) -> SemanticNode | EvalError:
    function_name = function.nodes[0]
    function_expr = function.nodes[1]

    if name.name == function_name.name:
        return function

    replaced_function_expr = replace_name_with_expr_in_expr(name, expr, function_expr)

    if type(replaced_function_expr) == EvalError:
        return replaced_function_expr

    return SemanticNode(SemanticNodeType.function, nodes=[function_name, replaced_function_expr])


def replace_name_with_expr_in_apply(name: SemanticNode, expr: SemanticNode, apply: SemanticNode) -> SemanticNode | EvalError:
    basics = apply.nodes

    replaced_basics = replace_name_with_expr_in_basics(name, expr, basics)

    if type(replaced_basics) == EvalError:
        return replaced_basics

    return SemanticNode(SemanticNodeType.apply, nodes=replaced_basics)


def replace_name_with_expr_in_basics(name: SemanticNode, expr: SemanticNode, basics: list[SemanticNode]) -> list[SemanticNode] | EvalError:
    replaced_basics = []

    for basic in basics:
        replaced_basic = replace_name_with_expr_in_basic(name, expr, basic)

        if type(replaced_basic) == EvalError:
            return replaced_basic

        replaced_basics.append(replaced_basic)

    return replaced_basics


def replace_name_with_expr_in_basic(name: SemanticNode, expr: SemanticNode, basic: SemanticNode) -> SemanticNode | EvalError:
    node = basic.nodes[0]

    match(node):
        case SemanticNodeType.integer:
            return node
        case SemanticNodeType.name:
            if node.name == name.name:
                return deepcopy(expr)
            else:
                return node
        case SemanticNodeType.expr:
            return replace_name_with_expr_in_expr(name, expr, node)
        case SemanticNodeType.lazy_record:
            return replace_name_with_expr_in_expr(name, expr, node)
        case SemanticNodeType.eager_record:
            return replace_name_with_expr_in_expr(name, expr, node)


def replace_name_with_expr_in_lazy_record(name: SemanticNode, expr: SemanticNode, lazy_record: SemanticNode) -> SemanticNode | EvalError:
    pairs = lazy_record.nodes

    replaced_pairs = replace_name_with_expr_in_pairs(name, expr, pairs)

    if type(replaced_pairs) == EvalError:
        return replaced_pairs

    return SemanticNode(SemanticNodeType.eager_record, nodes=replaced_pairs)


def replace_name_with_expr_in_eager_record(name: SemanticNode, expr: SemanticNode, eager_record: SemanticNode) -> SemanticNode | EvalError:
    pairs = eager_record.nodes

    replaced_pairs = replace_name_with_expr_in_pairs(name, expr, pairs)

    if type(replaced_pairs) == EvalError:
        return replaced_pairs

    return SemanticNode(SemanticNodeType.eager_record, nodes=replaced_pairs)


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
        print(tokens)
        exit(1)

    abstract_syntax_tree = syntax_parse_tokens(tokens)

    if type(abstract_syntax_tree) is SyntaxParseError:
        print(abstract_syntax_tree)
        exit(1)

    semantic_context = SemanticContext(bound_names=['plus', 'minus', 'mult', 'div', 'cond'])

    semantic_tree = semantic_parse_expr(abstract_syntax_tree, semantic_context)

    if type(semantic_tree) is SemanticParseError:
        print(semantic_tree)
        exit(1)

    result = eval_expr(semantic_tree)

    # print(semantic_tree)


if __name__ == "__main__":
    main()
