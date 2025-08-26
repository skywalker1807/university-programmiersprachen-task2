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
    integer = auto()  # [0-9]+
    identifier = auto()  # [a-zA-Z]+

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
        return f"Token{{literal={self.literal}, token_type={self._token_type}, location={self._source_code_location}}}"


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

    apply = auto()
    function = auto()  # kids: [name, expr]

    integer = auto()
    name = auto()

    lazy_record = auto()  # kids: list[pairs]
    eager_record = auto()  # kids: list[pairs]
    pair = auto()  # kids: list[pair]

    # syntactic elements
    equal_sign = "="
    comma = ","
    dot = "."

    # parenthesis
    left_parenthesis = "("
    right_parenthesis = ")"
    left_bracket = "["
    right_bracket = "]"
    left_brace = "{"
    right_brace = "}"


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
                case AbstractSyntaxTreeNodeType.integer:
                    string += f"{self.token.literal}"
                case AbstractSyntaxTreeNodeType.name:
                    string += f"{self.token.literal}"
                case AbstractSyntaxTreeNodeType.lazy_record:
                    string += f"{{{', '.join([node.pretty_string() for node in self.nodes])}}}"
                case AbstractSyntaxTreeNodeType.eager_record:
                    string += f"[{', '.join([node.pretty_string() for node in self.nodes])}]"
                case AbstractSyntaxTreeNodeType.pair:
                    string += f"{self.nodes[0].pretty_string()} = {self.nodes[1].pretty_string()}"
        return string


class ParseError(Exception):
    def __init__(self, message: str, nodes: AbstractSyntaxTreeNode | None = None, index: int | None = None, recoverable: bool = True, at_end=False) -> None:
        super().__init__(message)
        location = ""
        if nodes and index:
            self._message = f"{print_source_code_location(nodes, index)} - {message}"
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


def peek_node(nodes: list[AbstractSyntaxTreeNode], index: int) -> AbstractSyntaxTreeNode | None:
    return nodes[index] if index < len(nodes) else None


def print_source_code_location(nodes: list[AbstractSyntaxTreeNode], index: int) -> str:
    if index < 0:
        index = 0
    elif index >= len(nodes):
        index = len(nodes) - 1

    return f"{nodes[index].token.source_code.path}:{nodes[index].token.source_code_location}"


def advance_node(nodes: list[AbstractSyntaxTreeNode], index: int) -> tuple[AbstractSyntaxTreeNode | ParseError, int]:
    node = peek_node(nodes, index)
    if node is None:
        error = ParseError(
            f"unexpected end of input\nTODO: snippet",
            nodes=nodes,
            index=index,
            recoverable=False,
            at_end=True,
        )
        return error, index

    return node, index + 1


def match_node(nodes: list[AbstractSyntaxTreeNode], index: int, node_types: list[AbstractSyntaxTreeNodeType]) -> tuple[AbstractSyntaxTreeNode | None, int]:
    node = peek_node(nodes, index)

    if node is None:
        return None, index

    if node.node_type not in node_types:
        return None, index

    return node, index + 1


def expect_node(nodes: list[AbstractSyntaxTreeNode], index: int, node_types: list[AbstractSyntaxTreeNodeType]) -> tuple[AbstractSyntaxTreeNode | ParseError, int]:
    node = peek_node(nodes, index)
    if node is None:
        return ParseError(
            f"expected {', '.join(f"'{node_type}'" for node_type in node_types)}, found <end of input>\nTODO: snippet",
            nodes=nodes,
            index=index,
            recoverable=False,
            at_end=True,
        ), index

    if node.node_type not in node_types:
        error = ParseError(
            f"expected {', '.join(f"'{node_type}'" for node_type in node_types)} found {node.token.token_type}: {node.token.literal}\nTODO: snippet",
            nodes=nodes,
            index=index,
        )
        return error, index

    return node, index + 1


class Context:
    level: int


def parse_function(nodes: list[AbstractSyntaxTreeNode], index: int = 0) -> tuple[AbstractSyntaxTreeNode, int]:
    start_index = index

    first: AbstractSyntaxTreeNode = peek_node(nodes, index)
    second: AbstractSyntaxTreeNode = peek_node(nodes, index + 1)

    if not first or first.node_type != AbstractSyntaxTreeNodeType.name:
        error = ParseError(
            message=f"expected name in function: <name> . <expr>\nTODO: snippet",
            nodes=nodes,
            index=index,
            recoverable=True
        )
        return error, start_index

    if not second or second.node_type != AbstractSyntaxTreeNodeType.dot:
        error = ParseError(
            message=f"expected dot in function: <name> . <expr>\nTODO: snippet",
            nodes=nodes,
            index=index,
            recoverable=True,
        )
        return error, start_index

    name, index = advance_node(nodes, index)
    dot, index = advance_node(nodes, index)
    expr, index = parse_expr(nodes, index)

    if type(expr) is ParseError:
        if expr.recoverable:
            error = ParseError(
                message=f"expected expr in function: <name> . <expr>\nTODO: snippet",
                nodes=nodes,
                index=index,
                recoverable=False,
            )
            return error, start_index
        else:
            return expr, start_index

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.function, nodes=[name, expr]), index


def parse_expr(nodes: list[AbstractSyntaxTreeNode], index: int = 0) -> tuple[AbstractSyntaxTreeNode | ParseError, int]:
    start_index = index

    function, index = parse_function(nodes, index)

    if type(function) is AbstractSyntaxTreeNode:
        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.expr, nodes=[function]), index

    if type(function) is ParseError and not function.recoverable:
        return function, start_index

    apply, index = parse_apply(nodes, index)

    if type(apply) is ParseError:
        return apply, start_index

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.expr, nodes=[apply]), index


def parse_apply(nodes: list[AbstractSyntaxTreeNode], index: int = 0) -> tuple[AbstractSyntaxTreeNode | ParseError, int]:
    start_index = index

    basics: list[AbstractSyntaxTreeNode] = []

    first, index = parse_basic(nodes, index)

    if type(first) is ParseError:
        return first, start_index

    basics.append(first)

    while True:
        next_basic, index_2 = parse_basic(nodes, index)

        if type(next_basic) is ParseError:
            if not next_basic.recoverable:
                return next_basic, start_index
            else:
                break

        index = index_2

        basics.append(next_basic)

    return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.apply, nodes=basics), index


def parse_basic(nodes: list[AbstractSyntaxTreeNode], index: int = 0) -> tuple[AbstractSyntaxTreeNode | ParseError, int]:
    start_index = index

    node = peek_node(nodes, index)
    if node is None:
        error = ParseError(
            message=f"expected (int/name/(...)/{...}/[...]), found <end of input>\nTODO: snippet",
            nodes=nodes,
            index=index,
            recoverable=True,
            at_end=True,
        )
        return error, start_index

    if node.node_type == AbstractSyntaxTreeNodeType.integer:
        integer, index = advance_node(nodes, index)

        return integer, index

    if node.node_type == AbstractSyntaxTreeNodeType.name:
        name, index = advance_node(nodes, index)

        return name, index

    if node.node_type == AbstractSyntaxTreeNodeType.left_parenthesis:
        left_parenthesis, index = advance_node(nodes, index)

        next_node = peek_node(nodes, index)
        if not next_node:
            error = ParseError(
                f"expected apply or function, found <end of input>\nTODO: snippet",
                nodes=nodes,
                index=index,
                recoverable=False,
            )
            return error, start_index

        if next_node.node_type == AbstractSyntaxTreeNodeType.right_parenthesis:
            error = ParseError(
                f"expected  apply or function, found {next_node.node_type}\nTODO: snippet",
                nodes=nodes,
                index=index,
                recoverable=False,
            )
            return error, start_index

        expr, index = parse_expr(nodes, index)

        if type(expr) is ParseError:
            if expr.at_end and expr.recoverable:
                error = ParseError(
                    f"expected ')', found <end of input>",
                    nodes=nodes,
                    index=index,
                    recoverable=False,
                )
                return error, start_index
            return expr, start_index

        right_parenthesis, index = expect_node(nodes, index, node_types=[AbstractSyntaxTreeNodeType.right_parenthesis])

        if type(right_parenthesis) is ParseError:
            return right_parenthesis, start_index

        return expr, index

    if node.node_type == AbstractSyntaxTreeNodeType.left_brace:
        left_brace, index = advance_node(nodes, index)

        next_node = peek_node(nodes, index)

        if not next_node:
            error = ParseError(
                f"expected name or '}}', found <end of input>\nTODO: snippet",
                nodes=nodes,
                index=index,
                recoverable=False
            )
            return error, start_index

        if next_node.node_type == AbstractSyntaxTreeNodeType.right_brace:
            next_node, index = advance_node(nodes, index)

            return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.lazy_record, nodes=[]), index

        if next_node.node_type != AbstractSyntaxTreeNodeType.name:
            error = ParseError(
                f"expected name, found {next_node.node_type}\nTODO: snippet",
                nodes=nodes,
                index=index,
                recoverable=False,
            )
            return error, start_index

        pairs, index = parse_pairs(nodes, index)

        if type(pairs) is ParseError:
            return pairs, start_index

        right_brace, index = expect_node(nodes, index, node_types=[AbstractSyntaxTreeNodeType.right_brace])
        if type(right_brace) is ParseError:
            return right_brace, start_index

        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.lazy_record, nodes=pairs), index

    if node.node_type == AbstractSyntaxTreeNodeType.left_bracket:
        left_bracket, index = advance_node(nodes, index)

        next_node = peek_node(nodes, index)

        if not next_node:
            error = ParseError(
                f"expected name or ']', found <end of input>\nTODO: snippet",
                nodes=nodes,
                index=index,
                recoverable=False,
            )
            return error, start_index

        if next_node.node_type == AbstractSyntaxTreeNodeType.right_bracket:
            next_node, index = advance_node(nodes, index)

            return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.eager_record, nodes=[]), index

        if next_node.node_type != AbstractSyntaxTreeNodeType.name:
            error = ParseError(
                f"expected name, found {next_node.node_type}\nTODO: snippet",
                nodes=nodes,
                index=index,
                recoverable=False,
            )
            return error, start_index

        pairs, index = parse_pairs(nodes, index)

        if type(pairs) is ParseError:
            return pairs, start_index

        right_bracket, index = expect_node(nodes, index, node_types=[AbstractSyntaxTreeNodeType.right_bracket])
        if type(right_bracket) is ParseError:
            return right_bracket, start_index

        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.lazy_record, nodes=pairs), index

    if node.node_type in [AbstractSyntaxTreeNodeType.dot, AbstractSyntaxTreeNodeType.equal_sign]:
        error = ParseError(
            f"expected (int/name/(...)/{{...}}/[...]), found {node.node_type}\nTODO: snippet",
            nodes=nodes,
            index=index,
            recoverable=False,
        )
        return error, start_index

    error = ParseError(
        f"expected (int/name/(...)/{{...}}/[...]), found {node.node_type}\nTODO: snippet",
        nodes=nodes,
        index=index,
    )
    return error, start_index


def parse_pairs(nodes: list[AbstractSyntaxTreeNode], index: int = 0) -> tuple[list[AbstractSyntaxTreeNode] | ParseError, int]:
    start_index = index

    pairs: list[AbstractSyntaxTreeNode] = []

    def parse_one_pair(nodes: list[AbstractSyntaxTreeNode], index: int, context: Context = None) -> tuple[AbstractSyntaxTreeNode | ParseError, int]:
        start_index = index

        name, index = expect_node(nodes, index, node_types=[AbstractSyntaxTreeNodeType.name])

        if type(name) is ParseError:
            return name, start_index

        equal_sign, index = expect_node(nodes, index, node_types=[AbstractSyntaxTreeNodeType.equal_sign])

        if type(equal_sign) is ParseError:
            return equal_sign, start_index

        value, index = parse_expr(nodes, index)

        if type(value) is ParseError:
            return value, start_index

        return AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.pair, nodes=[name, value]), index

    first, index = parse_one_pair(nodes, index)

    if type(first) is ParseError:
        return first, start_index

    pairs.append(first)

    while True:
        comma, index_2 = match_node(nodes, index, node_types=[AbstractSyntaxTreeNodeType.comma])
        if not comma:
            break

        next_pair, index = parse_one_pair(nodes, index_2)

        if type(next_pair) is ParseError:
            return next_pair, start_index

        pairs.append(next_pair)

    return pairs, index


def parse_tokens(tokens: list[Token]) -> AbstractSyntaxTreeNode | ParseError:
    nodes: list[AbstractSyntaxTreeNode] = []

    for token in tokens:
        node: AbstractSyntaxTreeNode

        match token.token_type:
            case TokenType.integer:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.integer, token=token)
            case TokenType.identifier:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.name, token=token)

            case TokenType.equal_sign:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.equal_sign, token=token)
            case TokenType.comma:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.comma, token=token)
            case TokenType.dot:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.dot, token=token)

            case TokenType.left_parenthesis:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.left_parenthesis, token=token)
            case TokenType.right_parenthesis:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.right_parenthesis, token=token)
            case TokenType.left_bracket:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.left_bracket, token=token)
            case TokenType.right_bracket:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.right_bracket, token=token)
            case TokenType.left_brace:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.left_brace, token=token)
            case TokenType.right_brace:
                node = AbstractSyntaxTreeNode(AbstractSyntaxTreeNodeType.right_brace, token=token)

            case _:
                continue

        nodes.append(node)

        if not nodes:
            return ParseError("empty input", recoverable=False)

    abstract_syntax_tree, index = parse_expr(nodes, 0)

    if type(abstract_syntax_tree) is ParseError:
        return abstract_syntax_tree

    if index != len(nodes):
        error = ParseError(
            f"unexpected '{nodes[index].node_type}'\nTODO: snippet",
            nodes=nodes,
            index=index,
            recoverable=False,
        )
        return error

    return abstract_syntax_tree


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

    abstract_syntax_tree = parse_tokens(tokens)

    if type(abstract_syntax_tree) is ParseError:
        print(abstract_syntax_tree, file=sys.stderr)
        exit(1)

    print(abstract_syntax_tree)


if __name__ == "__main__":
    main()
