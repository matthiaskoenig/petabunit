"""Working with units in petab problem."""

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import libsbml
import numpy as np
import pandas as pd
import pint
import sbmlutils.io
from pint import Quantity, UnitRegistry
from pint.errors import DimensionalityError, UndefinedUnitError
from sbmlutils.io import read_sbml

import petab
from petabunit import log
from petabunit.console import console

logger = log.get_logger(__name__)
UdictType = Dict[str, Optional[str]]

_sbml_uids = [
    "ampere",
    "farad",
    "joule",
    "lux",
    "radian",
    "volt",
    "avogadro",
    "gram",
    "katal",
    "metre",
    "second",
    "watt",
    "becquerel",
    "gray",
    "kelvin",
    "mole",
    "siemens",
    "weber",
    "candela",
    "henry",
    "kilogram",
    "newton",
    "sievert",
    "coulomb",
    "hertz",
    "litre",
    "ohm",
    "steradian",
    "dimensionless",
    "item",
    "lumen",
    "pascal",
    "tesla",
]


def default_ureg() -> Tuple[UnitRegistry, UdictType]:
    """Default unit registry. """
    sbml_uid_dict: UdictType = {}
    ureg = pint.UnitRegistry()
    ureg.define("none = count")
    ureg.define("item = count")
    ureg.define("percent = 0.01*count")

    # add predefined units (SBML Level 2)
    for uid, unit_str in {
        "substance": "mole",
        "volume": "litre",
        "area": "meter^2",
        "length": "meter",
        "time": "second",
    }.items():
        ureg.define(f"{uid} = {unit_str}")

    # add SBML definitions
    for key in _sbml_uids:
        try:
            _ = ureg(key)
            sbml_uid_dict[key] = key
        except UndefinedUnitError:
            logger.debug(f"SBML unit kind can not be used in pint: '{key}'")
    return ureg, sbml_uid_dict


ureg, sbml_uid_dict = default_ureg()


class UnitsParser:
    """Parsing of PEtab unit information."""

    @classmethod
    def model_uid_dict(cls, model: libsbml.Model) -> UdictType:
        """Populate the model uid dict for lookup."""

        # map no units on dimensionless
        uid_dict: Dict[str, str] = {**sbml_uid_dict, "": None}

        udef: libsbml.UnitDefinition
        for udef in model.getListOfUnitDefinitions():
            uid = udef.getId()
            unit_str = cls.udef_to_str(udef)
            q = ureg(unit_str)
            try:
                # check if uid is existing unit registry definition (short name)
                q_uid = ureg(uid)
                if q_uid == q:
                    unit_str = uid
                else:
                    # incorrect meaning
                    logger.error(
                        f"SBML uid interpretation of '{uid}' does not match unit "
                        f"registry: '{uid} = {q} != {q_uid}'."
                    )
            except UndefinedUnitError:
                pass
            #     # add definition
            #     definition = f"{uid} = {unit_str}"
            #     ureg.define(definition)

            uid_dict[uid] = unit_str

        return uid_dict

    @classmethod
    def from_sbml_file(cls, source: Union[str, Path]) -> UdictType:
        """Get pint UnitsInformation for SBMLDocument."""
        doc: libsbml.SBMLDocument = read_sbml(source)
        return cls.from_sbml_doc(doc=doc)

    @classmethod
    def from_sbml_doc(cls, doc: libsbml.SBMLDocument) -> UdictType:
        """Get pint UnitsInformation for SBMLDocument."""
        model: libsbml.Model = doc.getModel()
        if not model:
            ValueError(f"No model found in SBMLDocument: {doc}")
        return cls.from_sbml_model(model)


    @classmethod
    def from_sbml_model(cls, model: libsbml.Model) -> UdictType:
        """Get UnitsInformation for SBML Model."""

        # create uid to unit mapping
        uid_dict: UdictType = cls.model_uid_dict(model)

        # add additional units
        udict: UdictType = {}

        # add time unit
        time_uid: str = model.getTimeUnits()
        if time_uid:
            udict["time"] = uid_dict[time_uid]

        # get all objects in model
        if not model.isPopulatedAllElementIdList():
            model.populateAllElementIdList()
        sid_list: libsbml.IdList = model.getAllElementIdList()

        for k in range(sid_list.size()):
            sid = sid_list.at(k)
            element: libsbml.SBase = model.getElementBySId(sid)
            if not element:
                continue

            # in case of reactions we have to derive units from the kinetic law
            if isinstance(element, libsbml.Reaction):
                if element.isSetKineticLaw():
                    element = element.getKineticLaw()
                else:
                    udict[sid] = None
                    continue

            # for species check if amount or concentration
            if isinstance(element, libsbml.Species):
                species: libsbml.Species = element
                substance_uid = species.getSubstanceUnits()

                if substance_uid:
                    if species.getHasOnlySubstanceUnits():
                        # store amount
                        udict[sid] = uid_dict[substance_uid]
                    else:
                        # store concentration
                        compartment: libsbml.Compartment = model.getCompartment(
                            element.getCompartment()
                        )
                        volume_uid = compartment.getUnits()
                        if substance_uid and volume_uid:
                            udict[sid] = f"({uid_dict[substance_uid]})/({uid_dict[volume_uid]})"
                        else:
                            logger.debug(f"volume unit missing for concentration: '{sid}'")
                            udict[sid] = None

            if isinstance(element, (libsbml.Compartment, libsbml.Parameter)):
                udict[sid] = uid_dict[element.getUnits()]

            # else:
            #     udef: libsbml.UnitDefinition = element.getDerivedUnitDefinition()
            #     if udef is None:
            #         continue
            #
            #     # find the correct unit definition
            #     uid: Optional[str] = None
            #     udef_test: libsbml.UnitDefinition
            #     for udef_test in model.getListOfUnitDefinitions():
            #         if libsbml.UnitDefinition.areIdentical(udef_test, udef):
            #             uid = udef_test.getId()
            #             break
            #
            #     if uid:
            #         udict[sid] = uid_dict[uid]
            #     else:
            #         logger.warning(
            #             f"DerivedUnit not in UnitDefinitions: "
            #             f"'{cls.udef_to_str(udef)}'"
            #         )
            #         udict[sid] = cls.udef_to_str(udef)

        return udict


    # abbreviation dictionary for string representation
    _units_abbreviation = {
        "kilogram": "kg",
        "meter": "m",
        "metre": "m",
        "second": "s",
        "hour": "hr",
        "dimensionless": "",
        "katal": "kat",
        "gram": "g",
    }

    @classmethod
    def udef_to_str(cls, udef: libsbml.UnitDefinition) -> str:
        """Format SBML unitDefinition as string.

        Units have the general format
            (multiplier * 10^scale *ukind)^exponent
            (m * 10^s *k)^e

        Returns the string "None" in case no UnitDefinition was provided.

        """
        if udef is None:
            return "None"

        # order the unit definition
        libsbml.UnitDefinition.reorder(udef)

        # collect formated nominators and denominators
        nom = []
        denom = []
        for u in udef.getListOfUnits():
            m = u.getMultiplier()
            s = u.getScale()
            e = u.getExponent()
            k = libsbml.UnitKind_toString(u.getKind())

            # get better name for unit
            k_str = cls._units_abbreviation.get(k, k)

            # (m * 10^s *k)^e

            # handle m
            if np.isclose(m, 1.0):
                m_str = ""
            else:
                m_str = str(m) + "*"

            if np.isclose(abs(e), 1.0):
                e_str = ""
            else:
                e_str = "^" + str(abs(e))

            # FIXME: handle unit prefixes;

            if np.isclose(s, 0.0):
                if not m_str and not e_str:
                    string = k_str
                else:
                    string = "({}{}{})".format(m_str, k_str, e_str)
            else:
                if e_str == "":
                    string = "({}10^{}*{})".format(m_str, s, k_str)
                else:
                    string = "(({}10^{}*{})^{})".format(m_str, s, k_str, e_str)

            # collect the terms
            if e >= 0.0:
                nom.append(string)
            else:
                denom.append(string)

        nom_str = " * ".join(nom)
        denom_str = " * ".join(denom)
        if len(denom) > 1:
            denom_str = f"({denom_str})"
        if (len(nom_str) > 0) and (len(denom_str) > 0):
            return f"{nom_str}/{denom_str}"
        if (len(nom_str) > 0) and (len(denom_str) == 0):
            return nom_str
        if (len(nom_str) == 0) and (len(denom_str) > 0):
            return f"1/{denom_str}"
        return ""


    # @staticmethod
    # def normalize_changes(
    #     changes: Dict[str, Quantity], uinfo: "UnitsInformation"
    # ) -> Dict[str, Quantity]:
    #     """Normalize all changes to units in given units dictionary.
    #
    #     This is a major helper function allowing to convert changes
    #     to the requested units.
    #     """
    #     Q_ = uinfo.ureg.Quantity
    #     changes_normed = {}
    #     for key, item in changes.items():
    #         if hasattr(item, "units"):
    #             try:
    #                 # convert to model units
    #                 item = item.to(uinfo[key])
    #             except DimensionalityError as err:
    #                 logger.error(
    #                     f"DimensionalityError "
    #                     f"'{key} = {item}'. Check that model "
    #                     f"units fit with changes units."
    #                     f"\n{err}"
    #                 )
    #                 raise err
    #             except KeyError as err:
    #                 logger.error(
    #                     f"KeyError: '{key}' does not exist in unit "
    #                     f"dictionary of model."
    #                 )
    #                 raise err
    #         else:
    #             item = Q_(item, uinfo[key])
    #             logger.warning(
    #                 f"No units provided, assuming dictionary units: {key} = {item}"
    #             )
    #         changes_normed[key] = item
    #
    #     return changes_normed



def units_for_petab_problem(problem: petab.Problem):
    """Resolve all units for a given problem.

    TODO: dictionary of units which are mapped on pint units for all objects.
    """
    pass

def unit_statistics_for_doc(doc: libsbml.SBMLDocument) -> Dict[str, Any]:
    """Return unit information for given model."""

    udict = UnitsParser.from_sbml_doc(doc)
    console.print(udict)

    model: libsbml.Model = doc.getModel()

    info: Dict[str, Any] = {
        "model_id": model.getId(),
        "n_objects": len(udict),
        "n_units": len([k for k, v in udict.items() if v]),
    }
    info["f_units"] = info["n_units"] / info["n_objects"]
    info["units"] = {k: v for k, v in udict.items() if v}

    return info


def unit_statistics(sbml_paths: Iterable[Path]) -> pd.DataFrame:
    """Create pandas data frame with unit information for models."""
    infos = []
    for p in sbml_paths:
        if not p.exists():
            logger.error(f"SBML does not exist: '{p}'")

        doc: libsbml.SBMLDocument = sbmlutils.io.read_sbml(source=p)
        info = unit_statistics_for_doc(doc=doc)
        infos.append(info)

    return pd.DataFrame(infos)


if __name__ == "__main__":

    from petabunit import EXAMPLES_DIR
    sbml_paths = [
        EXAMPLES_DIR / "Elowitz_Nature2000" / "model_Elowitz_Nature2000.xml",
        EXAMPLES_DIR / "enalapril_pbpk" / "enalapril_pbpk.xml",
        EXAMPLES_DIR / "simple_chain" / "simple_chain.xml",
        EXAMPLES_DIR / "simple_pk" / "simple_pk.xml",
    ]
    df = unit_statistics(sbml_paths=sbml_paths)
    console.rule(style="white")
    console.print(df)
    console.rule(style="white")

    # import libsbml
    #
    # from sbmlutils.factory import UnitDefinition
    # from sbmlutils.report.units import udef_to_string
    #
    #
    # doc: libsbml.SBMLDocument = libsbml.SBMLDocument()
    # model: libsbml.Model = doc.createModel()

    # example definitions

    # for key, definition, _, _ in [
    #     ("mmole_per_min", "mmole/min", "str", "mmol/min"),
    #     ("m3", "meter^3", "str", "m^3"),
    #     ("m3", "meter^3/second", "str", "m^3/s"),
    #     ("mM", "mmole/liter", "str", "mmol/l"),
    #     ("ml_per_s_kg", "ml/s/kg", "str", "ml/s/kg"),
    #     ("dimensionless", "dimensionless", "str", "dimensionless"),
    #     ("item", "item", "str", "item"),
    #     ("mM", "mmole/min", "latex", "\\frac{mmol}/{min}"),
    # ]:
    #     ud = UnitDefinition(key, definition=definition)
    #     udef: libsbml.UnitDefinition = ud.create_sbml(model=model)
    #
    #     console.rule(style="white")
    #     console.print(udef)
    #     console.print(udef_to_string(udef, format="str"))
