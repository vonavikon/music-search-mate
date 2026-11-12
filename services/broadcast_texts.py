"""Texts of the Nov 2026 release announcement (broadcast + EN/RU toggle).

Single source of truth: scripts/broadcast.py sends these, the `bc:` callback
in bot/handler.py swaps them in place. Kept in a dependency-free module so
the stdlib-only script can import it inside the container.
"""
BROADCAST_TEXTS = {
    "ru": ("Привет! Мы обновили алгоритм поиска, теперь он работает "
           "быстрее, точнее и лучше держит контекст."),
    "en": ("Hi! We've updated the search algorithm: it now works faster, "
           "more accurately, and keeps context better."),
}

# events-table marker: one row per user that already received the message;
# a rerun skips them, so an interrupted broadcast resumes without duplicates.
BROADCAST_MARK = "bc_2026_11"
