"""Named columns of the cleaned SDWPF dataset."""

FEATURE_COLUMNS = ("Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1",
                   "Pab2", "Pab3", "Prtv", "Patv", "Wsin", "Wcos")
FLAG_COLUMNS = ("m_missing", "m_imputed", "m_outlier", "f_fault",
                "f_curtail", "f_stuck", "f_farm")

from .masks import OFFICIAL_COLUMNS
