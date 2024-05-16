"""Working with units in SBML models."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import libsbml
import numpy as np
import pandas as pd
import pint
from pint import UnitRegistry
from pint.errors import UndefinedUnitError
import sbmlutils.io

from petabunit import log
from petabunit.console import console


logger = log.get_logger(__name__)
UdictType = Dict[str, Optional[str]]


class SBMLUnitParser:
    """Parser for SBML unit information."""

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

    _sbml_l2_defaults = {
        "substance": "mole",
        "volume": "litre",
        "area": "meter^2",
        "length": "meter",
        "time": "second",
    }

    @classmethod
    def sbml_unit_registry(cls) -> UnitRegistry:
        """Create unit registry for given SBML file."""
        ureg = pint.UnitRegistry()
        ureg.define("none = count")
        ureg.define("item = count")
        ureg.define("percent = 0.01*count")

        return ureg

    @classmethod
    def model_uid_dict(cls, model: libsbml.Model, ureg: UnitRegistry) -> UdictType:
        """Populate the model uid dict for lookup."""

        # add SBML definitions
        uid_dict: Dict[str, Optional[str]] = {k: k for k in cls._sbml_uids}
        # map no units on dimensionless
        uid_dict[""] = None

        # add predefined units (SBML Level 2)
        uids = {udef.getId() for udef in model.getListOfUnitDefinitions()}
        sbml_level: int = model.getLevel()
        if sbml_level == 2:
            for uid, unit_str in cls._sbml_l2_defaults.items():
                if uid not in uids:
                    ureg.define(f"{uid} = {unit_str}")

        def q_are_equivalent(q1: pint.Quantity, q2: pint.Quantity):
            ratio = (q1 / q2).to_base_units()
            return ratio.dimensionless and np.isclose(ratio.magnitude, 1)

        udef: libsbml.UnitDefinition
        for udef in model.getListOfUnitDefinitions():
            uid = udef.getId()
            unit_str = cls.udef_to_str(udef)

            try:
                q = ureg(unit_str)
                # check if uid is existing unit registry definition (short name)
                q_uid = ureg(uid)

                # check for equal definition
                if q_are_equivalent(q_uid, q):
                    unit_str = uid
                else:
                    # incorrect meaning
                    logger.error(
                        f"SBML uid interpretation of '{uid}' does not match unit "
                        f"registry: '{uid} = {q} != {q_uid}'."
                    )
            except UndefinedUnitError:
                pass
            except pint.errors.DefinitionSyntaxError as err:
                logger.error(f"DefinitionSyntaxError for '{uid=}': {unit_str} {err}")

            uid_dict[uid] = unit_str

        # console.print(f"{uid_dict=}")

        return uid_dict

    @classmethod
    def from_sbml_file(cls, source: Union[str, Path]) -> Tuple[UdictType, UnitRegistry]:
        """Get pint UnitsInformation for SBMLDocument."""
        doc: libsbml.SBMLDocument = read_sbml(source)
        return cls.from_sbml_doc(doc=doc)

    @classmethod
    def from_sbml_doc(cls, doc: libsbml.SBMLDocument) -> Tuple[UdictType, UnitRegistry]:
        """Get pint UnitsInformation for SBMLDocument."""
        model: libsbml.Model = doc.getModel()
        if not model:
            ValueError(f"No model found in SBMLDocument: {doc}")
        return cls.from_sbml_model(model)

    @classmethod
    def from_sbml_model(cls, model: libsbml.Model) -> Tuple[UdictType, UnitRegistry]:
        """Get UnitsInformation for SBML Model."""

        # create uid to unit mapping
        ureg = cls.sbml_unit_registry()
        uid_dict: UdictType = cls.model_uid_dict(model=model, ureg=ureg)

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
                            udict[sid] = (
                                f"({uid_dict[substance_uid]})/({uid_dict[volume_uid]})"
                            )
                        else:
                            logger.debug(
                                f"volume unit missing for concentration: '{sid}'"
                            )
                            udict[sid] = None

            if isinstance(element, (libsbml.Compartment, libsbml.Parameter)):
                udict[sid] = uid_dict[element.getUnits()]

        return udict, ureg

    # abbreviation dictionary for string representation
    _units_replacements: Dict[str, str] = {
        "kilogram": "kg",
        "meter": "m",
        "metre": "m",
        "litre": "liter",
        # "second": "s",
        "hour": "hr",
        # "dimensionless": "",
        # "katal": "kat",
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
            k_str = cls._units_replacements.get(k, k)

            # (m * 10^s *k)^e
            m_str = "" if np.isclose(m, 1.0) else str(m) + "*"
            e_str = "" if np.isclose(abs(e), 1.0) else f"^{str(abs(e))}"

            # FIXME: handle unit prefixes;
            if np.isclose(s, 0.0):
                if not m_str and not e_str:
                    string = k_str
                else:
                    string = "({}{}{})".format(m_str, k_str, e_str)
            else:
                if e_str:
                    string = "(({}10^{}*{}){})".format(m_str, s, k_str, e_str)
                else:
                    string = "({}10^{}*{})".format(m_str, s, k_str)

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

    @classmethod
    def unit_statistics_for_doc(cls, doc: libsbml.SBMLDocument) -> Dict[str, Any]:
        """Return unit information for given model."""
        model: libsbml.Model = doc.getModel()
        udict, ureg = cls.from_sbml_doc(doc)
        console.rule(model.getId(), style="white")
        # console.print(udict)
        # convert to Pint units
        udict_pint = {k: ureg(v) for k, v in udict.items() if v}

        info: Dict[str, Any] = {
            "model_id": model.getId(),
            "objects": len(udict),
            "units": len(udict_pint),
        }
        info["f_units"] = info["units"] / info["objects"]
        info["udefs"] = udict_pint

        return info

    @classmethod
    def unit_statistics(cls, sbml_paths: Iterable[Path]) -> pd.DataFrame:
        """Create pandas data frame with unit information for models."""
        infos = []
        for p in sbml_paths:
            if not p.exists():
                logger.error(f"SBML does not exist: '{p}'")

            doc: libsbml.SBMLDocument = sbmlutils.io.read_sbml(p)
            info = cls.unit_statistics_for_doc(doc=doc)
            infos.append(info)

        df = pd.DataFrame(infos)
        df.sort_values(by=["f_units", "objects"], ascending=[False, True], inplace=True)
        return df
