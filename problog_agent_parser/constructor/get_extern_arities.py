# get_extern_arities.py
import ast

_TARGET = {
    "problog_export_raw",
    "problog_export",
    "problog_export_nondet"
}

def _deco_id(deco: ast.AST):
    if not isinstance(deco, ast.Call):
        return None
    f = deco.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None

def extern_arities(source: str):
    """
    Return:
    {
      func_name: {
        "mode_spec": ["+term", "-int", ...], # from decorator
        "docstring": "...", # from function docstring
      },
      ...
    }
    """
    tree = ast.parse(source)
    out = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        func_name = node.name
        func_docs = ast.get_docstring(node)
        lineno = getattr(node, "lineno", None)

        best = None  # (decorator_name, mode_spec)
        for deco in node.decorator_list:
            name = _deco_id(deco)
            if name not in _TARGET:
                continue

            marks = []
            for a in deco.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    marks.append(a.value)
                else:
                    raise ValueError(
                        f"Unexpected arity type in decorator {name} of function {func_name}"
                    )

            # 一个函数通常只会有一个 problog_export* 装饰器
            # 如果真的多个，按“最后一个出现的”覆盖（更符合 python 直觉）
            best = (name, marks)

        if best and best[1]:
            deco_name, mode_spec = best
            out[func_name] = {
                "mode_spec": mode_spec,
                "docstring": func_docs,
            }

    return out
