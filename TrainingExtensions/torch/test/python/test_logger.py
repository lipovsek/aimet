# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import unittest
import logging

from aimet_common.utils import AimetLogger


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class UseLogger(unittest.TestCase):
    def test_log_areas(self):
        logger.info("            ")
        logger.info("Testing test_log_areas()")

        # Test the UTILS logger
        utils_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
        utils_logger.debug("Testing Debug")
        utils_logger.info("Testing Info")
        utils_logger.warning("Testing Warning")
        utils_logger.error("Testing Error")
        utils_logger.critical("Testing Critical")
        utils_logger.critical("**************************************** \n")

        # Test the QUANT logger
        quant_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
        quant_logger.debug("Testing Debug")
        quant_logger.info("Testing Info")
        quant_logger.warning("Testing Warning")
        quant_logger.error("Testing Error")
        quant_logger.critical("Testing Critical")
        quant_logger.critical("**************************************** \n")

        # Test the SVD logger
        svd_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("**************************************** \n")

    def test_setting_log_level(self):
        logger.info("*** Testing test_setting_log_level() *** \n")
        svd_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)

        # The default logging level for SVD defined in default_logging_config.json is used.
        logger.info(
            "Log at the default log level for SVD defined in default_logging_config.json"
        )
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("****************************************\n")

        # Change the default log level for SVD.
        # Only CRITICAL level logs will be logged.
        logger.info("Change SVD area's logging level to Critical")
        AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Svd, logging.CRITICAL)
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("****************************************\n")

        # Change the default log level for SVD.
        # All logs will be logged.
        logger.info("Change SVD area's logging level to Critical")
        AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Svd, logging.DEBUG)
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("****************************************\n")

    def test_setting_log_level_for_all_areas(self):
        logger.info("*** test_setting_log_level_for_all_areas() ***\n")

        svd_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)
        quant_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
        util_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
        test_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

        # The default logging level for all Log Areas defined in default_logging_config.json is used.
        logger.info(
            "Log at the default log level for  all Log Areas defined in default_logging_config.json"
        )
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")

        quant_logger.warning("Testing Warning")
        quant_logger.error("Testing Error")

        util_logger.critical("Testing Critical")
        util_logger.info("Testing Info")

        test_logger.critical("Testing Critical")
        test_logger.critical("****************************************\n")

        # Change the default log level for all areas
        # Only CRITICAL level logs will be logged.
        logger.info("Change the logging level for all Log Areas to WARNING")
        AimetLogger.set_level_for_all_areas(logging.WARNING)

        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")

        quant_logger.warning("Testing Warning")
        quant_logger.error("Testing Error")

        util_logger.critical("Testing Critical")
        util_logger.info("Testing Info")

        test_logger.critical("Testing Critical")
        test_logger.critical("****************************************\n")
