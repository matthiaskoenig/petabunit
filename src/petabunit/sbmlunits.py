"""Functions for getting SBML units."""
from petabunit.console import console


if __name__ == "__main__":
    import libsbml

    from sbmlutils.factory import UnitDefinition
    from sbmlutils.report.units import udef_to_string

    doc: libsbml.SBMLDocument = libsbml.SBMLDocument()
    model: libsbml.Model = doc.createModel()

    for key, definition, _, _ in [
        ("mmole_per_min", "mmole/min", "str", "mmol/min"),
        ("m3", "meter^3", "str", "m^3"),
        ("m3", "meter^3/second", "str", "m^3/s"),
        ("mM", "mmole/liter", "str", "mmol/l"),
        ("ml_per_s_kg", "ml/s/kg", "str", "ml/s/kg"),
        ("dimensionless", "dimensionless", "str", "dimensionless"),
        ("item", "item", "str", "item"),
        ("mM", "mmole/min", "latex", "\\frac{mmol}/{min}"),
    ]:
        ud = UnitDefinition(key, definition=definition)
        # ud = UnitDefinition("item")
        udef: libsbml.UnitDefinition = ud.create_sbml(model=model)

        console.rule(style="white")
        console.print(udef)
        console.print(udef_to_string(udef, format="str"))
