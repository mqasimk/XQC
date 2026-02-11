"""Automated documentation generation script for the XQC library.

This script uses Sphinx's ``sphinx-apidoc`` to automatically generate
reStructuredText files for the ``xqc`` package and then builds the HTML
documentation using ``sphinx-build``.

Run the script with::

    python docs/generate_docs.py

The generated HTML files will be placed in ``docs/_build``.
"""

import os
import sys
import subprocess

def run_command(command, cwd=None):
    """Run a shell command and raise an exception if it fails.

    Args:
        command (list[str]): Command and arguments to execute.
        cwd (str, optional): Working directory for the command.
    """
    print(f"Running command: {' '.join(command)}")
    subprocess.check_call(command, cwd=cwd)

def main():
    # Directory containing this script (the docs folder)
    docs_dir = os.path.abspath(os.path.dirname(__file__))
    # Path to the source package (xqc) relative to the docs folder
    src_dir = os.path.abspath(os.path.join(docs_dir, "..", "xqc"))

    # Step 1: Generate .rst files for the package using sphinx-apidoc
    # The ``-f`` flag forces overwriting existing files.
    run_command([
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-f",
        "-o",
        docs_dir,
        src_dir,
    ])

    # Step 2: Build the HTML documentation
    build_dir = os.path.join(docs_dir, "_build")
    run_command([
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        docs_dir,
        build_dir,
    ])

    print(f"Documentation built successfully in {build_dir}")

if __name__ == "__main__":
    main()
