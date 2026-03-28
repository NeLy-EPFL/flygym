from pathlib import Path
import mkdocs_gen_files

package = "flygym"
root = Path(__file__).resolve().parents[1]
src_root = root / "src"
src_pkg = src_root / package

nav = mkdocs_gen_files.Nav()

for path in sorted(src_pkg.rglob("*.py")):
    if path.name.startswith("_"):
        continue

    module_path = path.relative_to(src_root).with_suffix("")
    parts = list(module_path.parts)
    
    if parts[1] == "warp":
        # Don't build Warp refs on the Mac
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        continue

    ident = ".".join(parts)

    doc_path = Path("api_reference", *parts).with_suffix(".md")
    if path.name == "__init__.py":
        doc_path = Path("api_reference", *parts, "index.md")

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n::: {ident}\n")

# # Optional: only keep this if you actually use literate-nav (see below)
# with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
