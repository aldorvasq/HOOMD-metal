# Copyright (c) 2026 Aldo Vásquez-Briceño.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Convert a CUDA/HIP .cu file of simple kernels into a HOOMD-blue .metal file.

usage: cu2metal.py file.cu [-o file.metal]

The output is a starting point for the Metal port of the file (see hoomd/mtl/MetalKernel.h):

* Each __global__ kernel gets an argument struct (kernel_args::<kernel>_args) with one member
  per parameter, and a Metal kernel that copies the members into local variables named like
  the parameters, followed by the original body.
* Host driver functions are kept. hipLaunchKernelGGL calls become hoomd::mtl::launch calls that
  fill the argument struct, and hipFuncGetAttributes calls query the Metal pipeline.
* __device__ helper functions are copied into the Metal section.

The tool does not handle templated kernels, CUB/Thrust algorithms, warp intrinsics, or shared
memory. It prints a warning for each construct that needs attention. Always review the output.
"""

import argparse
import pathlib
import re
import sys

WARN = []


def warn(msg):
    WARN.append(msg)


def match_brace(text, start, open_ch="{", close_ch="}"):
    """Index just past the brace that closes the one at text[start]."""
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '"' or ch == "'":
            j = i + 1
            while j < len(text) and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if text.startswith("//", i):
            i = text.index("\n", i)
            continue
        if text.startswith("/*", i):
            i = text.index("*/", i) + 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def split_top(text, sep=","):
    out, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return [o.strip() for o in out if o.strip()]


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


class Param:
    def __init__(self, decl):
        decl = strip_comments(decl)
        decl = decl.split("=")[0].strip()
        decl = decl.replace("__restrict__", "").replace("__restrict", "")
        decl = " ".join(decl.split())
        m = re.match(r"(.*?)(\w+)\s*$", decl)
        self.type = m.group(1).strip()
        self.name = m.group(2)
        self.pointer = "*" in self.type
        if self.pointer:
            self.pointee = self.type[: self.type.rindex("*")].strip()
        if "&" in self.type:
            warn(f"reference parameter {decl}")

    def member(self):
        if self.pointer:
            return f"HOOMD_DEVPTR({self.pointee}) {self.name};"
        t = re.sub(r"^const\s+", "", self.type)
        return f"{t} {self.name};"

    def local(self):
        if self.pointer:
            return f"device {self.pointee}* {self.name} = args.{self.name};"
        return f"{self.type} {self.name} = args.{self.name};"


class Kernel:
    pass


def find_functions(text, keyword):
    """Find functions declared with a keyword (__global__ or __device__) at any position."""
    funcs = []
    for m in re.finditer(rf"\b{keyword}\b", text):
        # function start: preceding template<...> or line start
        line_start = text.rfind("\n", 0, m.start()) + 1
        start = line_start
        prev = text[:line_start].rstrip()
        if prev.endswith(">") and re.search(r"template\s*<[^;{}]*>\s*$", prev):
            start = re.search(r"template\s*<[^;{}]*>\s*$", prev).start()
        paren = text.index("(", m.end())
        close = match_brace(text, paren, "(", ")")
        brace = text.index("{", close)
        if ";" in text[close:brace]:
            continue  # declaration only
        end = match_brace(text, brace)
        head = text[m.end() : paren]
        name = re.search(r"(\w+)\s*$", head).group(1)
        funcs.append((start, end, name, text[paren + 1 : close - 1], text[brace:end], text[start:paren]))
    return funcs


def inside_class(text, pos):
    """True when position pos is inside a class or struct body (not only namespaces)."""
    stack = []
    for m in re.finditer(r"namespace\s+(\w+)\s*\{|\{|\}", text[:pos]):
        tok = m.group(0)
        if tok.startswith("namespace"):
            stack.append(m.group(1))
        elif tok == "{":
            stack.append(None)
        elif stack:
            stack.pop()
    return None in stack


def namespaces_at(text, pos):
    """Namespace stack enclosing position pos."""
    stack = []
    for m in re.finditer(r"namespace\s+(\w+)\s*\{|\{|\}", text[:pos]):
        tok = m.group(0)
        if tok.startswith("namespace"):
            stack.append(m.group(1))
        elif tok == "{":
            stack.append(None)
        else:
            if stack:
                stack.pop()
    return [s for s in stack if s is not None]


def qualify_shared_pointers(body):
    """Add the threadgroup address space to pointers derived from shared memory."""
    shared = set(re.findall(r"HIP_DYNAMIC_SHARED\(\s*[\w:<> ]+?\s*,\s*(\w+)\s*\)", body))
    shared |= set(re.findall(r"threadgroup\s+[\w:<> ]+?\*\s*(\w+)\s*=", body))
    shared |= set(re.findall(r"__shared__\s+[\w:<> ]+?\s+(\w+)\s*\[", body))
    # declarations start a line, and their type does not span lines (so comments are not types)
    decl = re.compile(
        r"(?m)^([ \t]*)((?:const[ \t]+)?[\w:<>]+(?:[ \t]+[\w:<>]+)*)[ \t]*\*[ \t]*(\w+)\s*=\s*([^;]+);"
    )
    changed = True
    while changed:
        changed = False

        def repl(m):
            nonlocal changed
            indent, t, name, expr = m.group(1), m.group(2), m.group(3), m.group(4)
            if t.startswith("threadgroup") or t.startswith("device") or t in ("return",):
                return m.group(0)
            if not any(re.search(rf"\b{v}\b", expr) for v in shared):
                return m.group(0)
            expr = re.sub(r"\((\s*(?:const\s+)?[\w:<>]+(?:\s+[\w:<>]+)*\s*)\*\s*\)",
                          r"(threadgroup \1*)", expr)
            shared.add(name)
            changed = True
            return f"{indent}threadgroup {t}* {name} = {expr};"

        body = decl.sub(repl, body)
    return body


MSL_SAFE_CUH = ("hoomd/ParticleData.cuh", "hoomd/BondedGroupData.cuh", "ParticleData.cuh",
                "BondedGroupData.cuh")


def convert(path):
    src = pathlib.Path(path).read_text()
    stem = pathlib.Path(path).stem

    for construct in ("thrust::", "hipcub::", "cub::", "__shfl", "__shared__", "HIP_DYNAMIC_SHARED",
                      "WarpReduce", "WarpScan", "__syncthreads", "atomic", "texture", "__ballot"):
        if construct in src:
            warn(f"uses {construct}")

    file_shared = re.findall(r"^extern\s+__shared__\s+([\w:<> ]+?)\s+(\w+)\s*\[\s*\]\s*;",
                             src, flags=re.M)
    kernels = find_functions(src, "__global__")
    device_funcs = []
    for f in find_functions(src, "__device__"):
        if inside_class(src, f[0]):
            # member functions of host functors (thrust, cub) stay in the host code
            warn(f"device member function {f[2]} of a class: port by hand if kernels use it")
        else:
            device_funcs.append(f)
    kernel_names = {k[2] for k in kernels}

    ks = []
    for start, end, name, params, body, head in kernels:
        k = Kernel()
        k.name = name
        k.params = [Param(p) for p in split_top(params)]
        k.body = body
        k.ns = namespaces_at(src, start)
        k.template = "template" in head
        if k.template:
            warn(f"templated kernel {name}: port by hand (function constants)")
        ks.append(k)

    if not ks:
        sys.exit(f"{path}: no kernels found")

    ns = [n for n in ks[0].ns if n != "kernel"]
    nsq = "::".join(ns)

    # host code: the source with kernels and device functions removed
    spans = sorted([(s, e) for s, e, *_ in kernels] + [(s, e) for s, e, *_ in device_funcs])
    host = []
    last = 0
    for s, e in spans:
        host.append(src[last:s])
        last = e
    host.append(src[last:])
    host = "".join(host)

    # rewrite launches
    def launch_repl(m):
        start = m.start()
        paren = m.end() - 1
        close = match_brace(host_text[0], paren, "(", ")")
        inner = host_text[0][paren + 1 : close - 1]
        parts = split_top(inner)
        kname = parts[0].strip("() ")
        kname = re.sub(r"HIP_KERNEL_NAME\((.*)\)", r"\1", kname).strip("() ")
        base = kname.split("<")[0].strip()
        if base not in kernel_names or "<" in kname:
            warn(f"launch of {kname} needs manual conversion")
            return None
        k = next(k for k in ks if k.name == base)
        grid, threads, shmem = parts[1], parts[2], parts[3]
        values = parts[5:]
        if len(values) != len(k.params):
            warn(f"launch of {base}: {len(values)} arguments for {len(k.params)} parameters")
            return None
        line_start = host_text[0].rfind("\n", 0, start) + 1
        indent = host_text[0][line_start:start]
        if indent.strip():
            indent = ""
        lines = ["{", f"{indent}    kernel_args::{base}_args _args;"]
        for p, v in zip(k.params, values):
            lines.append(f"{indent}    _args.{p.name} = {v};")
        lines.append(
            f'{indent}    hoomd::mtl::launch(HOOMD_MTL_SOURCE,\n{indent}                       '
            f'"{nsq}::{base}",\n{indent}                       {grid},\n'
            f"{indent}                       {threads},\n{indent}                       _args,\n"
            f"{indent}                       {shmem});"
        )
        lines.append(f"{indent}    }}")
        return start, close + (1 if host_text[0][close:close + 1] == ";" else 0), "\n".join(lines)

    # rewrite triple chevron launches as hipLaunchKernelGGL
    def chevron(text):
        while True:
            m = re.search(r"(\w+)\s*<<<([^;]*?)>>>\s*\(", text)
            if not m:
                return text
            close = match_brace(text, m.end() - 1, "(", ")")
            config = split_top(m.group(2))
            while len(config) < 4:
                config.append("0")
            inner = text[m.end() : close - 1]
            call = (f"hipLaunchKernelGGL(({m.group(1)}), {config[0]}, {config[1]}, {config[2]}, "
                    f"{config[3]}, {inner})")
            text = text[: m.start()] + call + text[close:]

    host = chevron(host)

    host_text = [host]
    while True:
        m = None
        for cand in re.finditer(r"hipLaunchKernelGGL\s*\(", host_text[0]):
            r = launch_repl(cand)
            if r:
                m = r
                break
        if not m:
            break
        s, e, rep = m
        host_text[0] = host_text[0][:s] + rep + host_text[0][e:]
    host = host_text[0]

    def attr_repl(m):
        kname = m.group(2).strip()
        kname = re.sub(r"^\(const void\s*\*\)\s*", "", kname)
        kname = re.sub(r"reinterpret_cast<const void\s*\*>\(&?(.*)\)", r"\1", kname)
        kname = kname.strip("&() ")
        return (
            f"hoomd::mtl::get_function_attributes(&{m.group(1)}, HOOMD_MTL_SOURCE, "
            f'"{nsq}::{kname}")'
        )

    host = re.sub(r"hipFuncGetAttributes\s*\(\s*&(\w+)\s*,\s*([^;]*?)\)\s*(?=;)", attr_repl, host)

    # includes, including those of the matching .cuh header
    includes = re.findall(r'^\s*#include\s+[<"]([^>"]+)[>"]', src, flags=re.M)
    cuh = pathlib.Path(path).with_suffix(".cuh")
    if cuh.exists():
        includes += re.findall(r'^\s*#include\s+[<"]([^>"]+)[>"]', cuh.read_text(), flags=re.M)
    shared_includes, host_includes = [], []
    src_dir = pathlib.Path(path).parent
    includes = [
        inc for inc in includes
        if inc.startswith(("hoomd/", "thrust", "hipcub", "cub", "cuda"))
        or (src_dir / inc).exists()
    ]
    for inc in includes:
        if (inc.endswith(".cuh") and inc not in MSL_SAFE_CUH) or inc.startswith(
            ("thrust", "hipcub", "cub", "cuda")
        ):
            host_includes.append(inc)
        elif inc in ("hip/hip_runtime.h", "assert.h", "stdio.h", "hoomd/TextureTools.h",
                     "hoomd/CachedAllocator.h", "CachedAllocator.h"):
            continue
        elif inc not in shared_includes:
            shared_includes.append(inc)
    # the host section re-includes its own includes from the original text; strip them
    host = re.sub(r'^\s*#include\s+[<"][^>"]+[>"]\s*\n', "", host, flags=re.M)
    host = re.sub(r"^\s*#pragma GCC diagnostic.*\n", "", host, flags=re.M)

    out = []
    out.append("// Copyright (c) 2009-2026 The Regents of the University of Michigan.")
    out.append("// Part of HOOMD-blue, released under the BSD 3-Clause License.\n")
    out.append(f"/*! \\file {stem}.metal\n    \\brief Metal port of {stem}.cu\n*/\n")
    out.append('#include "hoomd/mtl/MetalKernel.h"\n')
    for inc in shared_includes:
        out.append(f'#include "{inc}"')
    out.append("\n#ifndef __METAL_VERSION__")
    out.append(f'#include "{stem}.cuh"')
    for inc in host_includes:
        if not inc.endswith(f"{stem}.cuh") and not inc.startswith(("thrust", "hipcub", "cub", "cuda")):
            out.append(f'#include "{inc}"')
    out.append("#endif\n")

    # macros from the original (e.g. SMALL)
    macros = re.findall(r"^#(?:define|if|else|elif|endif)[^\n]*$", src[: src.find("namespace")], flags=re.M)
    macros = [m for m in macros if "_CUH" not in m and "_H__" not in m]

    for n in ns:
        out.append(f"namespace {n}\n    {{")
    out.append("namespace kernel_args\n    {")
    for k in ks:
        out.append(f"struct {k.name}_args\n    {{")
        for p in k.params:
            out.append(f"    {p.member()}")
        out.append("    };\n")
    out.append("    } // end namespace kernel_args\n")
    out.append("#ifdef __METAL_VERSION__\n")
    out.extend(macros)
    out.append("")
    for start, end, name, params, body, head in device_funcs:
        out.append(src[start:end] + "\n")
        warn(f"device function {name}: check pointer and reference address spaces")
    for k in ks:
        out.append(f"kernel void {k.name}(HOOMD_KERNEL_PARAMS(kernel_args::{k.name}_args))\n    {{")
        for p in k.params:
            out.append(f"    {p.local()}")
        out.append("")
        body = k.body[1:].lstrip("\n")
        for t, name in file_shared:
            declared = re.search(rf"(__shared__|HIP_DYNAMIC_SHARED)[^;]*\b{name}\b", body)
            if re.search(rf"\b{name}\b", body) and not declared:
                body = (f"    threadgroup {t}* {name} = (threadgroup {t}*)hoomd_dynamic_shared;\n"
                        + body)
        # dynamic shared memory
        body = re.sub(
            r"extern\s+__shared__\s+([\w:<> ]+?)\s+(\w+)\s*\[\s*\]\s*;",
            r"threadgroup \1* \2 = (threadgroup \1*)hoomd_dynamic_shared;",
            body,
        )
        body = qualify_shared_pointers(body)
        out.append(body)
        out.append("")
    out.append("#else\n")
    # host code: strip the namespace wrappers of the original and keep the kernel namespace
    first_ns = re.search(r"\bnamespace\s+\w+\s*\{", host)
    host_body = host[first_ns.start() :] if first_ns else host
    for n in ns:
        host_body = re.sub(rf"namespace\s+{n}\s*\{{", "", host_body, count=1)
    host_body = host_body.rstrip()
    for _ in ns:
        host_body = host_body[: host_body.rstrip().rfind("}")]
    out.append(host_body.strip())
    out.append("\n#endif\n")
    for n in reversed(ns):
        out.append(f"    }} // end namespace {n}")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    text = convert(args.source)
    out = args.output or str(pathlib.Path(args.source).with_suffix(".metal"))
    pathlib.Path(out).write_text(text)
    for w in WARN:
        print(f"{args.source}: {w}")


if __name__ == "__main__":
    main()
