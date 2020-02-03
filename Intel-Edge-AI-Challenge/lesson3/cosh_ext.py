# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for TensorFlow layer extractor for Model Optimizer
#
# You need to modify this file if you need several attributes of the layer
# to appear in the IR in different format than the default one. Then you
# need to implement pre-processing logic here.
#
# Refer to the section "Extending Model Optimizer with New Primitives" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op
from mo.front.tf.extractors.utils import *
from mo.front.common.partial_infer.utils import convert_tf_padding_to_str




class coshFrontExtractor(FrontExtractorOp):
    op = 'cosh'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.attr
        # extracting parameters from TensorFlow layer and prepare them for IR
        attrs = {
            'op': __class__.op
        }

        # update the attributes of the node
        Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)

        return __class__.enabled