"""Label set of the Uzbek news classification dataset (murodbek/uz-text-classification)."""

LABELS: list[str] = [
    "Avto",
    "Ayollar",
    "Dunyo",
    "Foto",
    "Iqtisodiyot",
    "Jamiyat",
    "Jinoyat",
    "Madaniyat",
    "O‘zbekiston",
    "Pazandachilik",
    "Qonunchilik",
    "Salomatlik",
    "Siyosat",
    "Sport",
    "Texnologiya",
]

# English names, handy for the README, plots and the demo.
LABELS_EN: dict[str, str] = {
    "Avto": "Auto",
    "Ayollar": "Women",
    "Dunyo": "World",
    "Foto": "Photo",
    "Iqtisodiyot": "Economy",
    "Jamiyat": "Society",
    "Jinoyat": "Crime",
    "Madaniyat": "Culture",
    "O‘zbekiston": "Uzbekistan",
    "Pazandachilik": "Cooking",
    "Qonunchilik": "Law",
    "Salomatlik": "Health",
    "Siyosat": "Politics",
    "Sport": "Sport",
    "Texnologiya": "Technology",
}

LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))
