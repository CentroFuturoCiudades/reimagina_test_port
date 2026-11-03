"""City presets, from exe_reimagina's ``src/pipeline/cities/*.toml``.

Source URIs point at the CFC data lake. See exe_reimagina ``doc/CITIES.md`` for
the meaning of each parameter.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CityPreset:
    name: str
    # Lot id city key: uuid5(NS, f"{key}:lot:{cadastral key}").
    key: str
    ent_code: str
    muni_code: str
    year: str
    # Metric CRS used for areas, distances and the height rasters.
    target_crs: str
    lots_uri: str
    # Cadastral columns joined with "-" into each lot's key.
    key_columns: tuple[str, ...]
    normativa_uri: str
    # Normativa source column -> internal name (zoning, max_cos, max_cus, ...).
    normativa_columns: dict[str, str]
    normativa_max_distance: float = 3.0
    # Open Buildings 2.5D temporal tiles and their manifests.
    tiles_uri: str = (
        "az://sandbox/projects/EXE/cleaned/open_buildings_temporal_data_v1_mx"
    )
    manifests_uri: str = (
        "az://sandbox/projects/EXE/raw/open_buildings_temporal_data_v1_mx"
    )
    presence_threshold: float = 0.5
    height_percentile: float = 95.0
    average_level_height: float = 2.7
    gsd_pixel: float = 0.5
    pois_uri_prefix: str = "az://sandbox/projects/EXE/cleaned/pois_accesibilidad"
    poi_max_distance_m: float = 20.0
    mf_min_units: float = 1.5

    @property
    def cve_code(self) -> str:
        return f"{self.ent_code}{self.muni_code}"

    @property
    def census_uri(self) -> str:
        return (
            "az://cleaned/inegi/censo/manzanas_2020.parquet/"
            f"entidad={int(self.ent_code)}/mun={int(self.muni_code)}"
        )

    @property
    def blocks_uri(self) -> str:
        return (
            "az://cleaned/inegi/marco_geostadistico/2024/manzanas_urbanas.parquet/"
            f"CVE_ENT={self.ent_code}/CVE_MUN={self.muni_code}"
        )

    @property
    def pois_uri(self) -> str:
        return f"{self.pois_uri_prefix}/pois_{self.cve_code}.geoparquet"


CITIES = {
    "mexicali": CityPreset(
        name="Mexicali",
        key="mexicali",
        ent_code="02",
        muni_code="002",
        year="2023",
        target_crs="EPSG:32611",
        lots_uri=(
            "az://raw/por_estado/02_baja_california/002_mexicali/imip/"
            "PRED_CAT_CP/PREDIOS_CAT_CP.shp"
        ),
        key_columns=("Numdel", "Colonia", "Manzana", "Lote", "Fraccion"),
        normativa_uri=(
            "az://sandbox/projects/EXE/raw/normativa/"
            "Predios (Normativa sintesis).geojson"
        ),
        normativa_columns={
            "USOSUELO24": "zoning",
            "Max_cos": "max_cos",
            "Max_cus": "max_cus",
            "Dens_U_max": "max_density",
        },
    ),
}
