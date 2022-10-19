# spacy_ud
Transforms Spacy English parser output to a Universal Dependency (UD) document

Spacy English parser produces a Spacy document

This code transforms SpacyDocs into UdDocs

Usage:
If you already have a Spacy document (output of Spacy), use this to create a UdDoc:

ud_doc = spacy_to_ud_doc(spacy_doc)

to print a UdDoc, use:

ud_doc.print_doc()

To create a Spacy doc, first create an instance of SpacyParser

spr = SpacyParser()

To get a Spacy doc, use

spacy_doc = spr.nlp('Spacy parser is pretty good')

To print a Spacy doc, use:

print_spacy_doc(spacy_doc)

For more information, see comments inside main.py
