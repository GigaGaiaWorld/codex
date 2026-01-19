from __future__ import annotations

from debug_toolkit.toolkit import BaseToolKit, CtxBinding, tool_card


class EchoToolKit(BaseToolKit, name="EchoKit", description="Echo toolkit"):
    prefix = CtxBinding(default="[")
    suffix = CtxBinding(default="]")

    @tool_card(expose=["message", "tag"])
    def echo(self, message: str, tag: str | None = None) -> dict[str, str | None]:
        """Echo a message with optional tag."""
        tag_value = f"<{tag}>" if tag is not None else None
        return {
            "result": f"{self.prefix}{message}{self.suffix}",
            "tag": tag_value,
        }


def main() -> None:
    toolkit = EchoToolKit({"prefix": "(", "suffix": ")"})

    print("--- call with explicit None for tag ---")
    result = toolkit.echo(message="hello", tag=None)
    print(result)

    print("--- call with tag value ---")
    result = toolkit.echo(message="hello", tag="greeting")
    print(result)


if __name__ == "__main__":
    main()
