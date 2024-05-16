"""Functionality for converting units."""


class PEtabUnitConverter:
    """Converter for units in PETab."""

    pass
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
