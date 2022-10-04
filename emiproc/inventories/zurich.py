
from os import PathLike
from pathlib import Path
from emiproc.inventories import Inventory


class MapLuftZurich(Inventory):

    mapluft_gdb: Path

    def __init__(self, mapluft_gdb: PathLike) -> None:
        self.mapluft_gdb = Path(mapluft_gdb)

        super().__init__()
