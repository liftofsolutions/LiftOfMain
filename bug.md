# Known bugs

## Compressed `.rar` files cannot be read when only bsdtar is installed

**Status:** open, deferred. A dependency issue, to be solved later.
**Found:** 2026-10-19, in a parse dry run of `Btest/data`.
**Where:** `src/documentProcessing/documentAnalyzer.py` (`openArchive`,
`configureRarTools`), through `rarfile` 4.5.

### Symptom

A `.rar` job fails with:

```
Nothing in 'Downloads.rar' could be read. Skipped
Downloads.rar/the_solar_system_an_overview.txt: Could not be unpacked:
BadRarFile: Failed the read enough data: req=7621 got=49
3 more were skipped too.
```

The archive opens and lists its files, but every member fails to unpack.
The same happens with the local copy and the copy on Drive (same bytes).

### The archive is fine

`Btest/data/Downloads.rar` is RAR5, not solid, not encrypted, normal
compression. Windows' own bsdtar (3.8.9) unpacks it correctly:

```
C:\Windows\System32\tar.exe -xOf Downloads.rar the_solar_system_an_overview.txt
```

### Root cause

rarfile picks the first tool it finds, in this order: `unrar`, `unar`,
`7z`, `7zz`, `bsdtar`. On this machine only bsdtar exists, and rarfile
4.5 builds a broken command for it:

```
tar -x --to-stdout -f -- Downloads.rar the_solar_system_an_overview.txt
```

`-f` takes `--` as the archive name, so bsdtar fails with
`tar: Error opening archive: Failed to open '--'`. Those 49 bytes of
error text are what rarfile read in place of the file.

rarfile reads files stored *uncompressed* in a rar by itself, without a
tool, so those still work. That is likely why earlier rar samples passed.

### Impact

- Windows: every compressed `.rar` fails.
- Linux: fails the same way if only `libarchive-tools` (bsdtar) is
  installed. Works if `unrar`, `unar` or 7-Zip is installed, since rarfile
  tries those first (not yet checked here).
- zip, tar, pdf, docx, txt and md are not affected.

### Also fragile

`configureRarTools` sets the tool to plain `tar`. On Windows, Python's
`shutil.which("tar")` finds Git's GNU tar on PATH, but the process Windows
starts is `C:\Windows\System32\tar.exe`, because Windows searches
System32 before PATH. It works only because of that search order.

### Fix options

1. **No code change:** install a tool rarfile tries before bsdtar.
   - 7-Zip: Linux package `7zip` (`7zz`); Windows 7-Zip, with `7z.exe` on
     PATH.
   - unrar: not free software, and often missing from AWS Linux
     repositories.
2. **Code change:** unpack `.rar` with bsdtar directly instead of through
   rarfile, calling Windows' bsdtar by its full path. bsdtar ships with
   Windows 10+ and comes with `libarchive-tools` on Linux.

### Reproduce

Submit a job whose link points to `Btest/data/Downloads.rar`, served over
HTTP, or call
`documentAnalyzer.analyzeDocument(Path("Downloads.rar"), "Downloads.rar")`
on a copy. Expect the job to end `failed` with the error above.

## PDF tables without grid lines are not counted

**Status:** open.
**Found:** 2026-10-19, in a parse dry run of `Btest/data`.
**Where:** `readPdf` in `src/documentProcessing/documentAnalyzer.py`,
`page.find_tables()` from pdfplumber.

### Symptom

`Btest/data/Our_Solar_System_Lithograph.pdf` is reported with
`tableCount=0`. Page 2 has a "FAST FACTS" table: Body, Equatorial Radius
(km, mi), Mean Distance from the Sun (km, mi, millions) and Moons, with 10
rows from Sun to Neptune.

### Cause

`find_tables()` by default builds tables from the lines drawn on the page.
This table has only 3 horizontal rules and no vertical ones, so nothing is
found. Tables without grid lines are common in typeset pdfs.

### Tried

Detecting tables from where the text sits instead
(`{"vertical_strategy": "text", "horizontal_strategy": "text"}`) finds 1
table on page 2, but also 1 on page 1, which has no table, only the planet
name labels. That trades a missed table for a false one. Whether the
table it finds on page 2 is really FAST FACTS was not checked.

## PDF text from multi-column pages is read across the columns

**Status:** open. Does not affect counts today; will affect chunking.
**Found:** 2026-10-19, in a parse dry run of `Btest/data`.
**Where:** `readPdf` in `src/documentProcessing/documentAnalyzer.py`,
`page.extract_text()` from pdfplumber.

### Symptom

Page 2 of `Our_Solar_System_Lithograph.pdf` is printed in three columns.
`extract_text()` reads straight across the page, so each line it returns
joins pieces of all three columns:

```
Humans have gazed at the heavens and tried to understand mea, and
Makemake — have similar compositions and are solid the two Voyagers exit
the Oort Cloud, a vast spherical shell of icy
```

### Impact

- Today: words and tokens are right, since the same words are there.
  `lineCount` (68) counts the joined lines, not the printed ones.
- Later: once pdf text is chunked and embedded, chunks will mix sentences
  from unrelated columns, and search will return jumbled passages.

### Fix options

1. Find the columns from the gaps in the words' x positions, crop the
   page into one piece per column (`page.crop`), and read them in order.
2. Use a layout-aware extractor, e.g. PyMuPDF's text blocks. Check its
   licence first: PyMuPDF is AGPL.

## How tables are used in the rag is not decided

**Status:** open. Next thing to implement.

### Today

Tables are only counted (`tableCount` for md, docx and pdf). Nothing
reads them as tables:

- md: a table's rows are plain text lines like any other.
- docx: each row's cells are joined as `a | b | c`, only to count words
  and tokens.
- pdf: the cells come out as plain text mixed in with the rest of the
  page, and tables without grid lines are not found (see above).

### To decide

- How a table's content is pulled out of each format: rows and cells, or
  plain text.
- How a table is chunked: whole, or in groups of rows, and whether the
  header row is repeated with each group so a chunk still says what its
  columns mean.
- How it is embedded and stored: as text (e.g. markdown) next to the
  page's other text, or as its own records with the table's metadata.
