"""Live validation of the artist-level fallback via MetadataService.resolve()."""
import asyncio
import os

from services.metadata import MetadataService


async def t(artist, track, video_id="TEST"):
    info = {
        "artist": artist,
        "track": track,
        "title": track,
        "channel": artist,
        "release_date": "",
    }
    key = os.environ.get("LASTFM_API_KEY", "")
    svc = MetadataService(key)
    m = await svc.resolve(info, video_id)
    await svc.aclose()
    print(f"--- {artist} - {track} ---")
    print(f"  artist : {m.artist!r}")
    print(f"  album  : {m.album!r}")
    print(f"  year   : {m.year!r}")
    print(f"  genre  : {m.genre!r}")
    bio = (m.bio[:160] + "...") if len(m.bio) > 160 else m.bio
    print(f"  bio    : {bio!r}")
    print(f"  artwork: {bool(m.artwork_path)}")
    print()


async def main():
    # Sparse track-level tags -> artist-level fallback expected.
    await t("Huey Lewis & The News", "Back in Time")
    # Regression: track.getInfo must still return genre+bio.
    await t("Michael Jackson", "Billie Jean")
    # Edge: niche/lesser-known artist -> graceful degradation (junk denylist).
    await t("Tame Impala", "The Less I Know The Better")


asyncio.run(main())
