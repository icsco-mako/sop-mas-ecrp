#!/bin/bash
# Compile both marked-up and clean versions of the paper
set -e

cd "$(dirname "$0")"

TEX="elsarticle-template-num-names"
STY="markup.sty"

enable_markup() {
    sed -i '' 's/^%\\markuptrue/\\markuptrue/' "$STY"
    sed -i '' 's/^\\markupfalse/%\\markupfalse/' "$STY"
}

disable_markup() {
    sed -i '' 's/^\\markuptrue/%\\markuptrue/' "$STY"
    sed -i '' 's/^%\\markupfalse/\\markupfalse/' "$STY"
}

echo "==> Compiling marked-up version..."
enable_markup
latexmk -pdf -interaction=nonstopmode "$TEX.tex" > /dev/null 2>&1
mv "$TEX.pdf" "$TEX-marked.pdf"
echo "    -> $TEX-marked.pdf"

echo "==> Compiling clean version..."
disable_markup
latexmk -pdf -interaction=nonstopmode "$TEX.tex" > /dev/null 2>&1
mv "$TEX.pdf" "$TEX-clean.pdf"
echo "    -> $TEX-clean.pdf"

echo "==> Restoring markuptrue as default..."
enable_markup

echo "Done!"
ls -lh "$TEX-marked.pdf" "$TEX-clean.pdf"
