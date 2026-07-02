"""Build tests/fixtures/tiny.epub — the smallest book that exercises every stage.

2 body chapters (summaries: 2 units), 1 section each (entities: 2 units + cross-chapter
alias clustering on "Eleanor Vance"/"Miss Vance"/"Eleanor" and "Tom"), ~2-3 chunks/chapter
(embeddings), one explicit date + an event sentence per chapter (dates/events). ~2.6 KB/ch.

Run:  python scripts/make_tiny_epub.py
"""
from __future__ import annotations

from pathlib import Path

from ebooklib import epub

CH1 = """\
On the third of October, 1887, a letter arrived at Blackwood Hall addressed to Miss Eleanor Vance. The envelope bore no return address, only a smear of red wax and the faint imprint of a crow. Eleanor turned it over twice before she dared to break the seal, for letters seldom came to the Hall, and never for her.

The house had stood above the village of Ashford for two hundred years, its chimneys black against the autumn sky. Tom, the stable boy, had carried the letter up from the post road himself, his boots still wet with river mud. "Came by the morning coach, miss," he said, and would not meet her eye.

Inside, the message was brief. Dr. Aldous Finch would arrive before nightfall, and Eleanor was to tell no one. She read the lines three times, then folded the paper into the pocket of her grey dress.

Dr. Finch came as the light was failing, a tall man stooped at the shoulders, his spectacles catching the lamplight as he climbed down from his carriage. "Miss Vance," he said, removing his hat. "I had hoped we might speak before the others learn I am here." They sat in the cold drawing room while Tom lit the fire, and the doctor spoke in a low voice of her late father, of debts she had never been told of, and of a name — Blackwood — older and stranger than the house that bore it.

That night Eleanor did not sleep. She stood at the east window watching the road to Ashford glimmer under a thin moon, certain that whatever had begun with the letter would not end at Blackwood Hall."""

CH2 = """\
They reached the River Ash on the fifth of October, 1887, two days after Eleanor Vance had left Blackwood Hall. The water ran high and brown after the autumn rains, and the old bridge at Marlowe had been carried away in the night. Tom led the horse down to the ford, where a flat-bottomed boat waited against the bank.

Captain Reyes was a broad, weathered man who had spent thirty years on rougher water than this. He looked Eleanor over once and shook his head. "Miss Vance, the current's wrong today," he said. "I've put three families across since dawn and I'll not pretend it's safe. If you must cross, you cross now, before the light goes."

Eleanor did not hesitate; she had not come so far from Ashford to turn back at a river. She paid the captain his fare, and Tom handed the bags down while Reyes worked the rope hand over hand against the pull of the Ash. Halfway across a log struck the side and the boat lurched, but Eleanor gripped the gunwale and did not cry out.

On the far bank the town of Marlowe rose grey and quiet. Captain Reyes pressed a folded paper into Eleanor's hand. "A man came asking after a Miss Vance this morning — tall, with spectacles. I told him I'd seen no one. Mind who you trust in Marlowe, miss." Then he pushed off into the current and was gone.

The name on the paper was one she knew — the same that Dr. Finch had spoken of at Blackwood. Eleanor closed her hand around it and turned toward the town, where the first lamps were showing in the windows. Tom followed a pace behind, leading the tired horse, and neither of them spoke of the spectacled man, nor of the long road back to Ashford that they both knew they might never take again."""


def _html(title: str, body: str) -> str:
    paras = "".join(f"<p>{p.strip()}</p>" for p in body.split("\n\n") if p.strip())
    return f"<h2>{title}</h2>{paras}"


def build(out_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("pagemind-tiny-fixture")
    book.set_title("The Blackwood Letter")
    book.set_language("en")
    book.add_author("A. Test Author")

    c1 = epub.EpubHtml(title="The Letter at Blackwood Hall", file_name="chap1.xhtml", lang="en")
    c1.content = _html("The Letter at Blackwood Hall", CH1)
    c2 = epub.EpubHtml(title="The River Crossing", file_name="chap2.xhtml", lang="en")
    c2.content = _html("The River Crossing", CH2)
    for c in (c1, c2):
        book.add_item(c)

    book.toc = (c1, c2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [c1, c2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(out_path), book)
    return out_path


if __name__ == "__main__":
    out = build(Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tiny.epub")
    print(f"wrote {out}")
