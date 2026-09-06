"""Parse SwDA into ordered (conversation, turn) records with dialogue-act labels.

SwDA text carries disfluency markup that must go before embedding:
  {F uh, } {C but } {D well }   discourse-marker / filler braces
  [ is, + is ]                  repair brackets (reparandum + repair)
  <<...>> <laughter>            non-verbal annotations
  /  --  #                      segmentation marks

act_tag needs normalizing too: `sd^e`, `qy^d`, `nn^e` etc. carry suffixes, and
some rows hold multiple comma-separated tags. Reduce to the base DAMSL tag.
"""

import glob
import os
import re

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "swda-master", "swda")
OUT = os.path.join(HERE, "swda_parsed.parquet")

BRACE = re.compile(r"\{[A-Z]\s*")
ANGLE = re.compile(r"<+[^>]*>+")
CURLY = re.compile(r"[{}]")
BRACK = re.compile(r"[\[\]\+]")
MULTI = re.compile(r"\s+")
DROP = re.compile(r"[#/]|--")


def clean(t):
    if not isinstance(t, str):
        return ""
    t = ANGLE.sub(" ", t)
    t = BRACE.sub(" ", t)
    t = CURLY.sub(" ", t)
    t = BRACK.sub(" ", t)
    t = DROP.sub(" ", t)
    t = t.replace("(( ", " ").replace(" ))", " ")
    return MULTI.sub(" ", t).strip()


def base_tag(tag):
    if not isinstance(tag, str):
        return "?"
    t = tag.split(",")[0].strip()
    t = t.split("^")[0].strip()
    t = t.replace("*", "").replace("@", "")
    return t if t else "?"


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "sw*utt", "*.csv")))
    print(f"utterance files: {len(files)}")
    rows = []
    for f in files:
        try:
            df = pl.read_csv(f, infer_schema_length=0)
        except Exception:
            continue
        need = {"act_tag", "caller", "text", "conversation_no", "transcript_index"}
        if not need <= set(df.columns):
            continue
        rows.append(
            df.select(
                [
                    pl.col("conversation_no").cast(pl.Utf8),
                    pl.col("transcript_index").cast(pl.Int64),
                    pl.col("caller").cast(pl.Utf8),
                    pl.col("act_tag").cast(pl.Utf8),
                    pl.col("text").cast(pl.Utf8),
                ]
            )
        )
    df = pl.concat(rows)
    df = df.with_columns(
        [
            pl.col("text").map_elements(clean, return_dtype=pl.Utf8).alias("clean"),
            pl.col("act_tag").map_elements(base_tag, return_dtype=pl.Utf8).alias("act"),
        ]
    )
    df = df.with_columns(pl.col("clean").str.split(" ").list.len().alias("ntok")).sort(
        ["conversation_no", "transcript_index"]
    )

    print(f"utterances: {df.height:,}  conversations: {df['conversation_no'].n_unique():,}")
    print(f"tokens/utterance: median {df['ntok'].median():.0f}  frac <=2 tok {(df['ntok'] <= 2).mean():.3f}")
    print("\ntop dialogue acts:")
    vc = df.group_by("act").len().sort("len", descending=True).head(14)
    for r in vc.iter_rows():
        print(f"  {r[0]:<8} {r[1]:>7,}  {r[1] / df.height:.3f}")

    print("\nagreement-relevant acts:")
    for a in ("aa", "ar", "ny", "nn", "b", "arp", "nd", "no"):
        n = df.filter(pl.col("act") == a).height
        if n:
            sub = df.filter(pl.col("act") == a)
            print(f"  {a:<5} n={n:>6,}  median tok {sub['ntok'].median():.0f}  e.g. {sub['clean'][0][:52]!r}")

    df.write_parquet(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
