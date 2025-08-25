#!/usr/bin/python3.12
import argparse
import sys
from pathlib import Path
from enum import StrEnum, auto
from copy import deepcopy


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

    # predefined identifier
    plus = auto()
    minus = auto()
    mult = auto()
    div = auto()
    cond = auto()

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
        return f"Token{{token_type={self._token_type}, location={self._source_code_location}}}"


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

                token_type: TokenType = None
                match source_code.content[location.start_index: location.end_index]:
                    case "plus":
                        token_type = TokenType.plus
                    case "minus":
                        token_type = TokenType.minus
                    case "mult":
                        token_type = TokenType.mult
                    case "div":
                        token_type = TokenType.div
                    case "cond":
                        token_type = TokenType.cond
                    case _:
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


class NodeType(StrEnum):
    expr = auto()

    apply = auto()
    function = auto()  # kids: [name, expr]

    integer = auto()
    name = auto()

    lazy_record = auto()  # kids: list[pairs]
    eager_record = auto()  # kids: list[pairs]
    pairs = auto()  # kids: list[pair]

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


class Node:
    def __init__(self, node_type: NodeType, nodes: list["Node"] | None = None, token: Token | None = None) -> None:
        self._node_type = node_type
        self._token = token
        self.nodes = nodes or []

    @property
    def node_type(self) -> NodeType:
        return self._node_type

    @property
    def token(self) -> Token | None:
        return self._token

    @property
    def nodes(self) -> list["Node"]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: list["Node"]) -> None:
        self._nodes = nodes

    def __str__(self) -> str:
        string = ""
        string += f"{self.node_type} = {{\n"
        if self._token:
            string += f"\t{self._token}\n"
        string += "}\n"
        return string


class ParseError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def peek_node(nodes: list[Node], index: int) -> Node | None:
    return nodes[index] if index < len(nodes) else None


def advance_node(nodes: list[Node], index: int) -> tuple[Node, int]:
    node = peek_node(nodes, index)
    if node is None:
        raise ParseError("unexpected end of input")

    return node, index + 1


def match_node(nodes: list[Node], index: int, node_type: NodeType) -> tuple[Node, int]:
    node = peek_node(nodes, index)

    if node is None:
        return None, index

    if node.node_type not in node_type:
        return None, index

    return node, index + 1


def expect_node(nodes: list[Node], index: int, *node_types: NodeType, context: str | None = None) -> tuple[Node, int]:
    node = peek_node(nodes, index)
    if node is None:
        raise ParseError(f"expected {', '.join(node_types)}, found <end of input>")

    if node.node_type not in node_types:
        raise ParseError(f"expected {', '.join(node_types)} found {node.node_type}")

    return node, index + 1


class Context:
    level: int


def parse_function(nodes: list[Node], index: int = 0, context: Context = None) -> tuple[Node, int]:
    print(" " * context.level + "function", index)


def parse_expr(nodes: list[Node], index: int = 0, context: Context = None) -> tuple[Node, int]:
    print(" " * context.level + "expr", index)
    context.level += 1
    first: Node = peek_node(nodes, index)
    second: Node = peek_node(nodes, index + 1)

    if first and first.node_type == NodeType.name and second and second.node_type == NodeType.dot:
        name, index = advance_node(nodes, index)
        dot, index = advance_node(nodes, index)
        expr, index = parse_expr(nodes, index, context=deepcopy(context))
        return Node(NodeType.expr, nodes=[name, dot, expr]), index

    apply, index = parse_apply(nodes, index, context=deepcopy(context))
    return Node(NodeType.expr, nodes=[apply]), index


def parse_apply(nodes: list[Node], index: int = 0, context: Context = None) -> tuple[Node, int]:
    print(" " * context.level + "apply", index)
    context.level += 1
    basics: list[Node] = []

    first, index = parse_basic(nodes, index, context=deepcopy(context))
    basics.append(first)

    while True:
        part_of_basic, _ = match_node(
            nodes, index, node_type=[NodeType.integer, NodeType.name, NodeType.left_parenthesis, NodeType.left_bracket, NodeType.left_brace]
        )
        if not part_of_basic:
            break
        next_basic, index = parse_basic(nodes, index, context=deepcopy(context))
        basics.append(next_basic)

    return Node(NodeType.apply, nodes=basics), index


def parse_basic(nodes: list[Node], index: int = 0, context: Context = None) -> tuple[Node, int]:
    print(" " * context.level + "basic", index)
    context.level += 1
    node = peek_node(nodes, index)
    if node is None:
        raise ParseError("expected (int/name/(...)/{...}/[...]), found <end of input>")

    if node.node_type == NodeType.integer:
        integer, index = advance_node(nodes, index)

        return Node(NodeType.basic, nodes=[integer]), index

    if node.node_type == NodeType.name:
        name, index = advance_node(nodes, index)

        return Node(NodeType.basic, nodes=[name]), index

    if node.node_type == NodeType.left_parenthesis:
        left_parenthesis, index = advance_node(nodes, index)
        expr, index = parse_expr(nodes, index, context=deepcopy(context))
        right_parenthesis, index = expect_node(nodes, index, NodeType.right_parenthesis, context=")")

        return Node(NodeType.basic, nodes=[left_parenthesis, expr, right_parenthesis]), index

    if node.node_type == NodeType.left_brace:
        left_brace, index = advance_node(nodes, index)

        pairs: Node | None = None
        next_node: Node = peek_node(nodes, index)
        if next_node and next_node.node_type != NodeType.right_brace:
            pairs, index = parse_pairs(nodes, index, context=deepcopy(context))

        right_brace, index = expect_node(nodes, index, NodeType.right_brace, context="}")

        kids = [left_brace] + ([pairs] if pairs else []) + [right_brace]

        return Node(NodeType.basic, nodes=kids), index

    if node.node_type == NodeType.left_bracket:
        left_bracket, index = advance_node(nodes, index)

        pairs: Node | None = None
        next_node: Node = peek_node(nodes, index)
        if next_node and next_node.node_type != NodeType.right_bracket:
            pairs, index = parse_pairs(nodes, index, context=deepcopy(context))

        right_bracket, index = expect_node(nodes, index, NodeType.right_bracket, context="]")

        kids = [left_bracket] + ([pairs] if pairs else []) + [right_bracket]

        return Node(NodeType.basic, nodes=kids), index

    raise ParseError(f"expected (int/name/(...)/{{...}}/[...]), found {node.node_type}")


def parse_pairs(nodes: list[Node], index: int = 0, context: Context = None) -> tuple[Node, int]:
    print(" " * context.level + "pairs", index)
    context.level += 1
    pairs: list[Node] = []

    def parse_one_pair(nodes: list[Node], index: int, context: Context = None) -> tuple[Node, int]:
        print(" " * context.level + "pair", index)
        context.level += 1
        name, index = expect_node(nodes, index, NodeType.name)

        equal_sign, index = expect_node(nodes, index, NodeType.equal_sign)

        value, index = parse_expr(nodes, index, context=deepcopy(context))

        return Node(NodeType.pair, nodes=[name, equal_sign, value]), index

    first, index = parse_one_pair(nodes, index, context=deepcopy(context))
    pairs.append(first)

    while True:
        comma, index_2 = match_node(nodes, index, NodeType.comma)
        if not comma:
            break
        next_pair, index = parse_one_pair(nodes, index_2, context=deepcopy(context))
        pairs.append(next_pair)

    return Node(NodeType.pairs, nodes=pairs), index


def parse(tokens: list[Token]) -> Node:
    nodes: list[Node] = []

    for token in tokens:
        node: Node

        match token.token_type:
            case TokenType.integer:
                node = Node(NodeType.integer, token=token)
            case TokenType.identifier:
                node = Node(NodeType.name, token=token)

            case TokenType.plus:
                node = Node(NodeType.name, token=token)
            case TokenType.minus:
                node = Node(NodeType.name, token=token)
            case TokenType.mult:
                node = Node(NodeType.name, token=token)
            case TokenType.div:
                node = Node(NodeType.name, token=token)
            case TokenType.cond:
                node = Node(NodeType.name, token=token)

            case TokenType.equal_sign:
                node = Node(NodeType.equal_sign, token=token)
            case TokenType.comma:
                node = Node(NodeType.comma, token=token)
            case TokenType.dot:
                node = Node(NodeType.dot, token=token)

            case TokenType.left_parenthesis:
                node = Node(NodeType.left_parenthesis, token=token)
            case TokenType.right_parenthesis:
                node = Node(NodeType.right_parenthesis, token=token)
            case TokenType.left_bracket:
                node = Node(NodeType.left_bracket, token=token)
            case TokenType.right_bracket:
                node = Node(NodeType.right_bracket, token=token)
            case TokenType.left_brace:
                node = Node(NodeType.left_brace, token=token)
            case TokenType.right_brace:
                node = Node(NodeType.right_brace, token=token)

            case _:
                continue

        nodes.append(node)

        if not nodes:
            raise ParseError("empty input")

    for node in nodes:
        print(node.token)

    context = Context()
    context.level = 0
    ast, index = parse_expr(nodes, 0, context=deepcopy(context))

    print(index)

    if index != len(nodes):
        extra: Node = nodes[index]
        extra_token: Token = extra.token
        raise ParseError(f"{extra_token.source_code.path}:{extra_token.source_code_location} - unexpected token after: {extra.node_type}\nTODO: snippet")

    return ast


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

    expression = parse(tokens)

    print(expression)


if __name__ == "__main__":
    main()
