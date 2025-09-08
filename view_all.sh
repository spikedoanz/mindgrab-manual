#/bin/sh
for f in *layer*; do
    echo ">>> Processing: $f"
    mrpeek -batch "$f"
done
