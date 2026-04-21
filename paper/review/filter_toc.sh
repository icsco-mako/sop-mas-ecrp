#!/bin/bash
# Filter subsubsection entries from .toc file
cd "$(dirname "$0")"
if [ -f response_letter.toc ]; then
    grep -v "subsubsection" response_letter.toc > response_letter.toc.tmp
    mv response_letter.toc.tmp response_letter.toc
fi
