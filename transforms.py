from typing import Tuple, List, Dict, Callable
import utils
import pandas as pd
import numpy as np
import re
import warnings
from collections import defaultdict
from statsmodels.tsa.filters.hp_filter import hpfilter


class Transforms:
    """
    Transforms

    AÑADIR VARIABLES A LAS TRANSFORMACIONES
    
    Basta con añadir una línea en la transformación correspondiente siguiendo el formato dado.
    Otra opción es añadirla en el método transform haciendo uso de los métodos predefinidos o creando nuevos.
    """

    # MOVING AVERAGE (para t = t+1, t+2, t-1, t-2)
    # var to; var from
    MOVING_AVG = [
        # (132, 14),
        # (133, 18),
        # (134, 21),
        # (135, 24),
    ]

    # Interpolate december to december
    INT_DEC2DEC = [
        (132, 14),
        (133, 18),
        (134, 21),
        (135, 24),
    ]

    # AGGREGATE SUM
    AGR_FOO_EXCLUDED_COUNTRY = ['indonesia', 'venezuela', 'filipinas']
    AGR_LAT_COUNTRY_ES = ['brasil', 'méxico', 'costa rica', 'el salvador', 'guatemala', 'honduras', 'nicaragua', 'panamá', 'república dominicana', 'argentina', 'chile', 'colombia', 'ecuador', 'paraguay', 'perú', 'uruguay', 'puerto rico']

    AGR_L = [24, 135]
    AGR_R = [
        # to, from, filter
        (25, 24, 'AGR-')
    ]
    AGR_MEDIAN = [
        5, 11, 12, 13, 14, 15, 16, 17, 24, 25, 30, 31, 32, 37, 
        38, 39, 44, 45, 46, 51, 53, 54, 56, 57, 65, 67, 109, 110, 111,
        129, 130, 131, 132, 135
    ]

    # TO REAL AND EUROS

    # to, from, h, op
    DF_TO_REAL = [
        (17, [15, 2], lambda x : x[0] - x[1] / 100, None),
    ]
    DF_TO_EUROS_L = [
        (24, [14, 69], lambda x : x[0] * x[1], None),
        (126, [103, 69], lambda x : x[0] / x[1], None),
        (127, [104, 69], lambda x : x[0] / x[1], None),
        (128, [108, 69], lambda x : x[0] / x[1], None),
        (129, [109, 69], lambda x : x[0] * x[1], None),
        (130, [110, 69], lambda x : x[0] * x[1], None),
        (131, [111, 69], lambda x : x[0] * x[1], None),
        (141, [65, 69], lambda x : x[0] * x[1], None),
        (142, [67, 69], lambda x : x[0] * x[1], None),
        (150, [63, 69], lambda x : x[0] / x[1], None),
        (151, [62, 69], lambda x : x[0] / x[1], None),
        (152, [106, 69], lambda x : x[0] / x[1], None),
        (153, [66, 69], lambda x : x[0] / x[1], None),
        (154, [107, 69], lambda x : x[0] / x[1], None),
        (155, [68, 69], lambda x : x[0] / x[1], None),
        (157, [156, 69], lambda x : x[0] / x[1], None),
    ]
    # var in ratio, var in level
    DF_TO_EUROS_R = [
        (25, 24, None),
    ]

    # GET ROE RATIOS
    DF_ROE_R = [
        (57, 56, None),
    ]

    # TO MILLIONS
    DF_TO_MILLIONS = [
        (18, [18], lambda x : x[0] / 1000, None),
        (21, [21], lambda x : x[0] / 1000, None),
        (26, [26], lambda x : x[0] / 1000, None),
        (28, [28], lambda x : x[0] / 1000, None),
        (62, [62], lambda x : x[0] / 1000, None),
        (63, [63], lambda x : x[0] / 1000, None),
        (64, [64], lambda x : x[0] / 1000, None),
        (66, [66], lambda x : x[0] / 1000, None),
        (68, [68], lambda x : x[0] / 1000, None),
        (95, [95], lambda x : x[0] / 1000, None),
        (97, [97], lambda x : x[0] / 1000, None),
        (103, [103], lambda x : x[0] / 1000, None),
        (104, [104], lambda x : x[0] / 1000, None),
        (105, [105], lambda x : x[0] / 1000, None),
        (106, [106], lambda x : x[0] / 1000, None),
        (107, [107], lambda x : x[0] / 1000, None),
        (108, [108], lambda x : x[0] / 1000, None),
        (156, [156], lambda x : x[0] / 1000, None),
    ]

    # GET HPc
    # PREMIUMS_EXCLUDED_CODES = ['vida', 'total', 'no vida']
    # COMPETITORS_EXCLUDED_CODES = ['mapfre', 'mercado']
    # HP filter lambda
    HP_LAMBDA = 177
    # var_num; lp value (None to use mercado or mapfre), number to use that number; var_level_num; HP variable
    HP_VARS_FULL = [
        (15, None, 14, 16),
    ]
    # same as HP_VARS_FULL but in the simplified version their cp and lp are not saved
    HP_VARS_SIMPLIFIED = [
        (31, 0, 30, 32),
        (38, 0, 37, 39),
        (56, "ROE", 56, None),
        (45, 0, 44, 46),
        (15, 0, 14, 16),
    ]

    # GET RATIO COMBINADO
    # var_num_rate; var_level_num;; var_hp_num; nums of levels that form the combined
    RC_VARS = [
        (45, 44, 46, (30, 37)),
    ]

    # GET PREVISIONES CUOTAS
    # cuotas, tasas crecimiento
    PREV_CUOTAS = [
        (5, 15)
    ]

    # ALERTS
    ALERTS = [
        # ALERTAS DE G2
        (
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [9,10], 
            lambda x: (x[0] < x[1]).astype(float), 
            None
        ),
        # ALERTAS DE CRECIMIENTO
        (
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [15,15], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [15,15], 
            lambda x: (x[0] < x[1]).astype(float),
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [15,22], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [15,2], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {}
            ]
        ),
        # ALERTAS DE RATIOS
        (
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [32,32], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [39,39], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [46,46], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [30,30], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [31,31], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [37,37], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [38,38], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [44,44], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [45,45], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [32,34], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [31,34], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [39,43], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [46,50], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [30,33], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [37,42], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [38,43], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [44,49], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [45,50], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),
        # ALERTA DE BENEFICIO Y RENTABILIDAD
        (
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [51,51], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [51,52], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [65,65], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [65,153], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [67,67], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [67,155], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [56,61], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [9,10], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),
        # ALERTAS DE COMPARACION
        (
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [15,19], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [31,36], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [30,35], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [38,41], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [37,40], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [45,48], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [44,47], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [51,100], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [65,152], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [67,154], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [56,102], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [109,126], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [110,127], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [111,128], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),
        # ALERTAS NUEVAS
        (
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [132,133], 
            lambda x: (x[0] != x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [132,134], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [132,132], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [31,31], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [30,30], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [32,32], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [38,38], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [37,37], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [39,39], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [45,45], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [44,44], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [46,46], 
            lambda x: (x[0] > x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [56,56], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [56,56], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [51,51], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [65,65], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [67,67], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'ME'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [109,151], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [110,150], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [109,109], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [110,110], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [111,111], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MJ'}
            ]
        ),(
            lambda s: 'ALERTA_' + '_&_'.join(s), 
            [111,157], 
            lambda x: (x[0] < x[1]).astype(float), 
            [
                {'competitor_code': 'MAP'},
                {'competitor_code': 'MAP'}
            ]
        )
    ]

    def __init__(self, df_info_dict: pd.DataFrame, df: pd.DataFrame, simplified:bool = False):
        self.df_info_dict = df_info_dict
        self.df = df
        self.simplified = simplified

    def transform(self):
        # DUPLICATES WARNING ARE NORMAL FOR HISTORICAL VARIABLES (in each scenario de historical are also included)

        # get set of all scenarios in data
        scenarios = set([k.split('_')[12][1:] for k in self.df.columns if 'cp' in k or 'lp' in k])

        # change to float
        self.df = self.df.replace('--', np.nan).replace('-', np.nan).astype(float)
        # interpolate data
        self.df = self._interpolate()
        # HFM to millions
        print(f"\nHFM to millions")
        self.df = utils.concat_replace(self.df, self._lambda_transform(scenario=None, var_info=self.DF_TO_MILLIONS), warn_duplicates=False)
        
        for s in scenarios:
            # get HPC
            print(f"\nGet HPC {s}")
            #self.df = utils.concat_replace(self.df, self._get_hpc(scenario=s, var_info=self.HP_VARS_FULL, simplified=False))
            self.df = utils.concat_replace(self.df, self._get_hpc(scenario=s, var_info=self.HP_VARS_SIMPLIFIED, simplified=self.simplified))
            # get ratio combinado
            print(f"\nGet Ratio Comb {s}")
            #self.df = utils.concat_replace(self.df, self._get_ratio_combinado(scenario=s))
            # get cp and lp cuotas
            print(f"\nGet Cuotas Previsiones {s}")
            self.df = utils.concat_replace(self.df, self._calculate_cuotas_prevision(scenario=s, var_info=self.PREV_CUOTAS))
            # to real and to euros
            print(f"\nTo Euros and Real {s}")
            self.df = utils.concat_replace(self.df, self._lambda_transform(scenario=s, var_info=self.DF_TO_REAL + self.DF_TO_EUROS_L))
        
        # calculate ratios
        for s in scenarios:
            # get ratios ROE
            self.df = utils.concat_replace(self.df, self._calculate_rate(scenario=s, var_info=self.DF_ROE_R))
            # get ratios euro
            self.df = utils.concat_replace(self.df, self._calculate_rate(scenario=s, var_info=self.DF_TO_EUROS_R))

        for s in scenarios:
            # get moving average
            self.df = utils.concat_replace(self.df, self._get_moving_average(scenario=s, var_info=self.MOVING_AVG, window=5))
            # interpolate december to december
            self.df = utils.concat_replace(self.df, self._interpolate_dec2dec(scenario=s, var_info=self.INT_DEC2DEC))

        # aggregate
        print("\nAggregate sum levels")
        df_agr = self._aggregate_sum(self.AGR_L)
        print("\nAggregate median")
        df_med = self._aggregate_median(self.AGR_MEDIAN)
        self.df = utils.concat_replace(self.df, df_agr)
        for s in scenarios:
            print("\nAggregate sum ratios")
            self.df = utils.concat_replace(self.df, self._calculate_rate(scenario=s, var_info=self.AGR_R))
            self.df = utils.concat_replace(self.df, df_med)

        for s in scenarios:
            # get alerts
            print(f"\nGet alerts {s}")
            self.df = utils.concat_replace(self.df, self._lambda_transform(scenario=s, var_info=self.ALERTS))

        return self.df

    def _interpolate(self):
        # interpolate data (forward and then backward)
        cp_filter = self.df.columns.str.contains('cp')
        lp_filter = self.df.columns.str.contains('lp')
        # get filtered data
        hist_data = utils.filter_data(self.df.iloc[:,(~cp_filter) & (~lp_filter)], 'hist')\
            .interpolate(method='linear', limit_direction='forward', limit_area=None)\
            .interpolate(method='linear', limit_direction='backward', limit_area=None)
        cp_data = utils.filter_data(self.df.iloc[:,cp_filter], 'cp')\
            .interpolate(method='linear', limit_direction='forward', limit_area=None)\
            .interpolate(method='linear', limit_direction='backward', limit_area=None)
        lp_data = utils.filter_data(self.df.iloc[:,lp_filter], 'lp')\
            .interpolate(method='linear', limit_direction='forward', limit_area=None)\
            .interpolate(method='linear', limit_direction='backward', limit_area=None)
        # hist_data = utils.filter_data(self.df.iloc[:,(~cp_filter) & (~lp_filter)], 'hist').interpolate(method='linear', fill_value="extrapolate", limit_direction='both', limit_area=None)
        # cp_data = utils.filter_data(self.df.iloc[:,cp_filter], 'cp').interpolate(method='linear', fill_value="extrapolate", limit_direction='both', limit_area=None)
        # lp_data = utils.filter_data(self.df.iloc[:,lp_filter], 'lp').interpolate(method='linear', fill_value="extrapolate", limit_direction='both', limit_area=None)
        # concat data
        return pd.concat([
            hist_data,
            cp_data,
            lp_data,
        ], axis=1, copy=False)


    def _calculate_rate(self, scenario, var_info):
        df_info_var = self.df_info_dict['variables']
        df = self.df
        v_periods = utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS']

        data = []
        for var_to, var_from, filter in var_info:
            t = df_info_var.loc[var_to]
            f = df_info_var.loc[var_from]

            # dictionary of parameter positions
            dict_params = {k:idx for idx, k in enumerate(f[v_periods['hist']].split('_')) if '$' in k}

            v_filter = utils.get_variable_name(
                f[v_periods['hist']],
                scenario_code=scenario,
                default='.*',
            )
            pattern = re.compile(v_filter.replace('\\', '\\\\'))
            cols_from = [k for k in df.columns if pattern.match(k)]
            if filter is not None:
                cols_from = [k for k in cols_from if filter in k]
            cols_from = np.asarray(cols_from)
            cols_from_split = [k.split('_') for k in cols_from]

            for c,c_split in zip(cols_from, cols_from_split):
                # hist, cp, lp names
                f_cp_name = utils.get_variable_name(
                    f[v_periods['cp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                f_lp_name = utils.get_variable_name(
                    f[v_periods['lp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_hist_name = utils.get_variable_name(
                    t[v_periods['hist']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_cp_name = utils.get_variable_name(
                    t[v_periods['cp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_lp_name = utils.get_variable_name(
                    t[v_periods['lp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )

                # calculate hist
                l = utils.filter_data(df[c], 'hist')[c]
                r = utils.filter_data(l / l.shift(4) - 1, 'hist')
                r.columns = [t_hist_name]
                data.append(r)
                
                # calculate cp
                if f_cp_name in df:
                    l = pd.concat([utils.filter_data(df[c], 'hist')[c],utils.filter_data(df[f_cp_name], 'cp')[f_cp_name]])
                    r = utils.filter_data(l / l.shift(4) - 1, 'cp')
                    r.columns = [t_cp_name]
                    data.append(r)

                    # calculate lp
                    if f_lp_name in df:
                        l = pd.concat([utils.filter_data(df[f_cp_name], 'cp')[f_cp_name],utils.filter_data(df[f_lp_name], 'lp')[f_lp_name]])
                        r = utils.filter_data(l / l.shift(4) - 1, 'lp')
                        r.columns = [t_lp_name]
                        data.append(r)
        return data

    def _calculate_cuotas_prevision(self, scenario, var_info):
        df_info_var = self.df_info_dict['variables']
        df = self.df
        v_periods = utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS']

        data = []
        for var_to, var_from in var_info:
            t = df_info_var.loc[var_to]
            f = df_info_var.loc[var_from]

            # dictionary of parameter positions
            dict_params = {k:idx for idx, k in enumerate(f[v_periods['hist']].split('_')) if '$' in k}

            v_filter = utils.get_variable_name(
                t[v_periods['hist']],
                scenario_code=scenario,
                default='.*',
            )
            pattern = re.compile(v_filter.replace('\\', '\\\\'))
            cols_from = [k for k in df.columns if pattern.match(k)]

            cols_from = np.asarray(cols_from)
            cols_from_split = [k.split('_') for k in cols_from]

            for c,c_split in zip(cols_from, cols_from_split):
                # hist, cp, lp names
                f_cp_name_map = utils.get_variable_name(
                    f[v_periods['cp']], 
                    scenario_code=scenario, 
                    competitor_code='MAP',
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                f_lp_name_map = utils.get_variable_name(
                    f[v_periods['lp']], 
                    scenario_code=scenario, 
                    competitor_code='MAP',
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                f_cp_name_me = utils.get_variable_name(
                    f[v_periods['cp']], 
                    scenario_code=scenario, 
                    competitor_code='ME',
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                f_lp_name_me = utils.get_variable_name(
                    f[v_periods['lp']], 
                    scenario_code=scenario, 
                    competitor_code='ME',
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_hist_name = utils.get_variable_name(
                    t[v_periods['hist']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_cp_name = utils.get_variable_name(
                    t[v_periods['cp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_lp_name = utils.get_variable_name(
                    t[v_periods['lp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )

                # calculate cp
                if not (t_hist_name in df and f_cp_name_map in df and f_cp_name_me in df):
                    continue

                rates = utils.filter_data(df[f_cp_name_map], 'cp')[f_cp_name_map] - utils.filter_data(df[f_cp_name_me], 'cp')[f_cp_name_me]
                levels = utils.filter_data(df[t_hist_name], 'hist')[t_hist_name][-4:].to_list()
                for idx,r in enumerate(rates):
                    levels.append(levels[idx] * (1+r))
                # delete 4 values from hist
                levels = levels[4:]
                data.append(utils.filter_data(pd.DataFrame(data=levels, columns=[t_cp_name], index=rates.index.tolist()), 'cp'))

                # calculate lp
                if not(f_lp_name_map in df and f_lp_name_me in df):
                    continue
                
                levels = levels[-4:]
                rates = utils.filter_data(df[f_lp_name_map], 'lp')[f_lp_name_map] - utils.filter_data(df[f_lp_name_me], 'lp')[f_lp_name_me]
                for idx,r in enumerate(rates):
                    levels.append(levels[idx] * (1+r))
                # delete 4 values from hist
                levels = levels[4:]
                data.append(utils.filter_data(pd.DataFrame(data=levels, columns=[t_lp_name], index=rates.index.tolist()), 'lp'))
                
        return data

    def _aggregate_sum(self, var_nums):
        df_info_var = self.df_info_dict['variables']
        dict_countries = utils.get_country_codes_dict(self.df_info_dict)
        df = self.df

        data = []
        for v in var_nums:
            v_to = df_info_var.loc[v]
            for p_name, period in utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS'].items():
                v_p_to = v_to[period]

                # columns from
                v_filter = utils.get_variable_name(
                    v_p_to,
                    default='.*'
                )
                pattern = re.compile(v_filter.replace('\\', '\\\\'))
                cols = [k for k in df.columns if pattern.match(k)]
                df_data = df[cols]

                # vectorize
                df_to = df_data.reset_index().melt(id_vars=['index'])
                cols = df_to.columns

                if df_to.empty:
                    continue

                # separate label into multiple columns
                c = df_to['variable'].str.split('_', expand=True)
                c.columns = ['origen', 'var', 'premium', 'country', 'competitor', 'region n', 'region g','trans', 'tipology', 'curr', 'nom', 't', 'scenario']
                df_to[c.columns] = c
                df_to = df_to.drop(columns=['variable'])
                df_to = df_to[~df_to['country'].isin([dict_countries[k] for k in self.AGR_FOO_EXCLUDED_COUNTRY])]
                df_to_I = df_to[df_to['country'].isin([dict_countries[k] for k in self.AGR_LAT_COUNTRY_ES])]

                # aggregate region n
                s = df_to.groupby([k for k in df_to.columns if k not in {'value', 'country', 'region g'}]).sum(min_count=1)
                s = s.reset_index()
                s['country'] = 'AGR-' + s['region n']
                s['region g'] = '\\'
                s['variable'] = s[c.columns].agg('_'.join, axis=1)

                # un-vectorize
                unmelted = s[cols].pivot(index=['index'], columns=['variable'])
                unmelted.columns = unmelted.columns.droplevel()
                data.append(unmelted)

                # aggregate region n FOO (all except indonesia)
                s = df_to.groupby([k for k in df_to.columns if k not in {'value', 'country', 'region n', 'region g'}]).sum(min_count=1)
                s = s.reset_index()
                s['country'] = 'AGR-FOO'
                s['region n'] = '\\'
                s['region g'] = '\\'
                s['variable'] = s[c.columns].agg('_'.join, axis=1)

                # un-vectorize
                unmelted = s[cols].pivot(index=['index'], columns=['variable'])
                unmelted.columns = unmelted.columns.droplevel()
                data.append(unmelted)

                # aggregate region LAT
                s = df_to_I.groupby([k for k in df_to_I.columns if k not in {'value', 'country', 'region n', 'region g'}]).sum(min_count=1)
                s = s.reset_index()
                s['country'] = 'AGR-LAT'
                s['region n'] = '\\'
                s['region g'] = '\\'
                s['variable'] = s[c.columns].agg('_'.join, axis=1)

                # un-vectorize
                unmelted = s[cols].pivot(index=['index'], columns=['variable'])
                unmelted.columns = unmelted.columns.droplevel()
                data.append(unmelted)

                # aggregate region g
                s = df_to.groupby([k for k in df_to.columns if k not in {'value', 'country', 'region n'}]).sum(min_count=1)
                s = s.reset_index()
                s['country'] = 'AGR-' + s['region g']
                s['region n'] = '\\'
                s['variable'] = s[c.columns].agg('_'.join, axis=1)

                # un-vectorize
                unmelted = s[cols].pivot(index=['index'], columns=['variable'])
                unmelted.columns = unmelted.columns.droplevel()
                unmelted = utils.filter_data(unmelted, p_name)
                data.append(unmelted)
        return data
                
    def _aggregate_median(self, var_nums):
        df_info_var = self.df_info_dict['variables']
        df = self.df
        dict_competitors = utils.get_competitor_dict(self.df_info_dict['j Competitor Codes'], lower=True)

        data = []
        for v in var_nums:
            v_to = df_info_var.loc[v]
            for p_name, period in utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS'].items():
                v_p_to = v_to[period]
                if not isinstance(v_p_to, str):
                    continue

                # columns from
                v_filter = utils.get_variable_name(
                    v_p_to,
                    default='.*'
                )
                pattern = re.compile(v_filter.replace('\\', '\\\\'))
                cols = [k for k in df.columns if pattern.match(k)]
                df_data = df[cols]

                # vectorize
                df_to = df_data.reset_index().melt(id_vars=['index'])
                cols = df_to.columns

                if df_to.empty:
                    continue

                # separate label into multiple columns
                c = df_to['variable'].str.split('_', expand=True)
                c.columns = ['origen', 'var', 'premium', 'country', 'competitor', 'region n', 'region g','trans', 'tipology', 'curr', 'nom', 't', 'scenario']
                df_to[c.columns] = c
                df_to = df_to.drop(columns=['variable'])

                # aggregate competitors
                s = df_to[df_to['competitor'] != dict_competitors['mercado']].groupby([k for k in df_to.columns if k not in {'value', 'competitor'}]).median()
                s = s.reset_index()
                s['competitor'] = dict_competitors['mediana competidores']
                s['variable'] = s[c.columns].agg('_'.join, axis=1)

                # un-vectorize
                unmelted = s[cols].pivot(index=['index'], columns=['variable'])
                unmelted.columns = unmelted.columns.droplevel()
                unmelted = utils.filter_data(unmelted, p_name)
                data.append(unmelted)
        return data

    def _lambda_transform(self, scenario:str, var_info: List[Tuple[int, List[int], Callable, List[Dict[str,str]]]]):
        """
        Applies a function to the given rows

        :param str scenario: scenario used
        :param List[Tuple[int, List[int], Callable, List[Dict[str,str]]]] var_info: list of functions to apply with their given information.
            Each should have a destination, list of variables used, function to apply, filter used for the variables if any
            The destination can be given as an id or as a function of the names of the variables used
        """
        df_info_var = self.df_info_dict['variables']
        df = self.df

        if scenario is None:
            scenario = '.*'

        data = []
        for to, id_list, op, filters in var_info:
            if filters is None:
                filters = [{}] * len(id_list)

            v_list = [df_info_var.loc[k] for k in id_list]
            if isinstance(to, int):
                v_to = df_info_var.loc[to]
            else:
                v_to = None

            for period_type, period in utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS'].items():
                v_name_list = [k[period] for k in v_list]
                if v_to is not None:
                    v_name_to = v_to[period]
                    if not isinstance(v_name_to, str):
                        continue

                if not all([isinstance(k, str) for k in v_name_list]):
                    continue

                # dictionary of parameter positions
                dict_params = {k:idx for idx, k in enumerate(v_name_list[0].split('_')) if '$' in k}

                # columns from
                # f = defaultdict(lambda: '.*')
                # f.update(filters[0])
                v_filter = utils.get_variable_name(
                    v_name_list[0],
                    scenario_code=scenario,
                    default='.*',
                    **filters[0],
                )
                pattern = re.compile(v_filter.replace('\\', '\\\\'))
                cols_from = np.asarray([k for k in df.columns if pattern.match(k)])
                cols_from_split = [k.split('_') for k in cols_from]

                for c, c_split in zip(cols_from, cols_from_split):
                    # get other variables with same parameters
                    c_names = [str(c)]
                    for f,n in zip(filters[1:], v_name_list[1:]):
                        d = {k:c_split[v] for k,v in dict_params.items()}
                        # d.update(f)
                        c_names.append(utils.get_variable_name(
                            n, 
                            scenario_code=scenario, 
                            d=d,
                            **f,
                        ))
                    # check all variables exist in data
                    if not all([(k in df) for k in c_names]):
                        continue

                    if v_to is not None:
                        n = utils.get_variable_name(
                            v_name_to, 
                            scenario_code=scenario, 
                            d={k:c_split[v] for k,v in dict_params.items()},
                        )
                    else:
                        n = to(c_names)
                        
                    ret = pd.DataFrame(op([df[k] for k in c_names]), columns=[n])
                    data.append(utils.filter_data(ret, type=period_type))

        return data

    def _interpolate_dec2dec(self, scenario:str, var_info, window:int = 5):
        df_info_var = self.df_info_dict['variables']
        df = self.df
        v_periods = utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS']

        data = []
        for var_to, var_from in var_info:
            t = df_info_var.loc[var_to]
            f = df_info_var.loc[var_from]

            # dictionary of parameter positions
            dict_params = {k:idx for idx, k in enumerate(f[v_periods['hist']].split('_')) if '$' in k}

            v_filter = utils.get_variable_name(
                f[v_periods['hist']],
                scenario_code=scenario,
                default='.*',
            )
            pattern = re.compile(v_filter.replace('\\', '\\\\'))
            cols_from = [k for k in df.columns if pattern.match(k)]
            cols_from = np.asarray(cols_from)
            cols_from_split = [k.split('_') for k in cols_from]

            for c,c_split in zip(cols_from, cols_from_split):
                # hist, cp, lp names
                f_cp_name = utils.get_variable_name(
                    f[v_periods['cp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                f_lp_name = utils.get_variable_name(
                    f[v_periods['lp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_hist_name = utils.get_variable_name(
                    t[v_periods['hist']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_cp_name = utils.get_variable_name(
                    t[v_periods['cp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_lp_name = utils.get_variable_name(
                    t[v_periods['lp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )

                # concatenate the whole series
                f_hist_series = utils.filter_data(df[c], 'hist').iloc[:,0]

                cp_index = utils.filter_data(df[[k for k in df.columns if 'cp' in k][0]], 'cp').index
                f_cp_series = utils.filter_data(df.get(f_cp_name, pd.DataFrame(data=[np.nan] * len(cp_index), index=cp_index)), 'cp').iloc[:,0]

                lp_index = utils.filter_data(df[[k for k in df.columns if 'lp' in k][0]], 'lp').index
                f_lp_series = utils.filter_data(df.get(f_lp_name, pd.DataFrame(data=[np.nan] * len(lp_index), index=lp_index)), 'lp').iloc[:,0]

                f_full = pd.concat([f_hist_series, f_cp_series, f_lp_series])

                # interpolate dec to dec
                f_is_december = pd.to_datetime(f_full.iloc[3:].reset_index()['index']).dt.month != 12
                f_full.iloc[3:].iloc[f_is_december.to_numpy()] = np.nan
                f_full = f_full.interpolate()

                # append to data
                ma_hist = utils.filter_data(f_full, 'hist')
                ma_hist.columns = [t_hist_name]
                if isinstance(t_hist_name, str):
                    data.append(ma_hist)
                ma_cp = utils.filter_data(f_full, 'cp')
                ma_cp.columns = [t_cp_name]
                if isinstance(t_cp_name, str):
                    data.append(ma_cp)
                ma_lp = utils.filter_data(f_full, 'lp')
                ma_lp.columns = [t_lp_name]
                if isinstance(t_lp_name, str):
                    data.append(ma_lp)

        return data

    def _get_moving_average(self, scenario:str, var_info, window:int = 5):
        df_info_var = self.df_info_dict['variables']
        df = self.df
        v_periods = utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS']

        data = []
        for var_to, var_from in var_info:
            t = df_info_var.loc[var_to]
            f = df_info_var.loc[var_from]

            # dictionary of parameter positions
            dict_params = {k:idx for idx, k in enumerate(f[v_periods['hist']].split('_')) if '$' in k}

            v_filter = utils.get_variable_name(
                f[v_periods['hist']],
                scenario_code=scenario,
                default='.*',
            )
            pattern = re.compile(v_filter.replace('\\', '\\\\'))
            cols_from = [k for k in df.columns if pattern.match(k)]
            cols_from = np.asarray(cols_from)
            cols_from_split = [k.split('_') for k in cols_from]

            for c,c_split in zip(cols_from, cols_from_split):
                # hist, cp, lp names
                f_cp_name = utils.get_variable_name(
                    f[v_periods['cp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                f_lp_name = utils.get_variable_name(
                    f[v_periods['lp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_hist_name = utils.get_variable_name(
                    t[v_periods['hist']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_cp_name = utils.get_variable_name(
                    t[v_periods['cp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )
                t_lp_name = utils.get_variable_name(
                    t[v_periods['lp']], 
                    scenario_code=scenario, 
                    d={k:c_split[v] for k,v in dict_params.items()}
                )

                # concatenate the whole series
                f_hist_series = utils.filter_data(df[c], 'hist').iloc[:,0]

                cp_index = utils.filter_data(df[[k for k in df.columns if 'cp' in k][0]], 'cp').index
                f_cp_series = utils.filter_data(df.get(f_cp_name, pd.DataFrame(data=[np.nan] * len(cp_index), index=cp_index)), 'cp').iloc[:,0]

                lp_index = utils.filter_data(df[[k for k in df.columns if 'lp' in k][0]], 'lp').index
                f_lp_series = utils.filter_data(df.get(f_lp_name, pd.DataFrame(data=[np.nan] * len(lp_index), index=lp_index)), 'lp').iloc[:,0]

                f_full = pd.concat([f_hist_series, f_cp_series, f_lp_series])

                # calculate ma
                ma = f_full.rolling(window=window, min_periods=1, center=True).mean()
                ma = pd.DataFrame(ma)

                # append to data
                ma_hist = utils.filter_data(ma, 'hist')
                ma_hist.columns = [t_hist_name]
                if isinstance(t_hist_name, str):
                    data.append(ma_hist)
                ma_cp = utils.filter_data(ma, 'cp')
                ma_cp.columns = [t_cp_name]
                if isinstance(t_cp_name, str):
                    data.append(ma_cp)
                ma_lp = utils.filter_data(ma, 'lp')
                ma_lp.columns = [t_lp_name]
                if isinstance(t_lp_name, str):
                    data.append(ma_lp)

        return data

    def _get_hpc(self, scenario:str, var_info, simplified:bool=False):
        """
        Calculates forecasts using the hp filter

        :param bool simplified: whether to save the cp and lp forecasts for the variable (hp is always saved), defaults to False
        """
        df_info_var = self.df_info_dict['variables']
        df = self.df
        dict_periods = utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS']
        dict_competitors = utils.get_competitor_dict(self.df_info_dict['j Competitor Codes'], lower=True)
        dict_premium = utils.get_premium_codes_dict(self.df_info_dict)
        dict_countries = utils.get_country_codes_dict(self.df_info_dict)
        dict_region_n = utils.get_country_to_region_dict(self.df_info_dict)
        dict_region_g = utils.get_country_to_region_g_dict(self.df_info_dict)
        # premiums_excluded = [dict_premium[k] for k in self.PREMIUMS_EXCLUDED_CODES]
        # comp_excluded = [dict_competitors[k] for k in self.COMPETITORS_EXCLUDED_CODES]

        data = []
        for v_num, h_last_var, v_level_num, hp_num in var_info:
            # get rate variable
            v = df_info_var.loc[v_num]
            v_hist_name = v[dict_periods['hist']]
            v_periods = [
                v[dict_periods[k]] for k in ['cp', 'lp']
            ]

            # get hp variable
            if hp_num is not None:
                v_hp = df_info_var.loc[hp_num]
                v_periods_hp = [
                    v_hp[dict_periods[k]] for k in ['hist', 'cp', 'lp']
                ]

            # get level variable
            v_level = df_info_var.loc[v_level_num]
            v_periods_level = [
                v_level[dict_periods[k]] for k in ['hist', 'cp', 'lp']
            ]

            # dictionary of parameter positions
            dict_params = {k:idx for idx, k in enumerate(v_hist_name.split('_')) if '$' in k}

            # columns from
            v_filter = utils.get_variable_name(
                v_hist_name,
                scenario_code=scenario,
                default='.*',
            )
            pattern = re.compile(v_filter.replace('\\', '\\\\'))
            hist_cols = np.asarray([k for k in df.columns if pattern.match(k)])
            hist_cols_split = [k.split('_') for k in hist_cols]

            # dictionary from historic to cp and lp
            dict_from_to = {
                c:[utils.get_variable_name(
                    p, 
                    scenario_code=scenario,
                    d={k:c_split[v] for k,v in dict_params.items()}
                ) for p in v_periods]
                for c,c_split in zip(hist_cols, hist_cols_split)
            }

            # do for those where the historic variable exists (if not, no interpolation can be done)
            for c,c_split in zip(hist_cols, hist_cols_split):
                # get historic variable
                v_hist = utils.filter_data(df[c], 'hist')[c]
                # there is already data use that one
                exists = False
                if dict_from_to[c][0] in df and dict_from_to[c][1] in df:
                    exists = True
                    v_fut = pd.concat([utils.filter_data(df[dict_from_to[c][0]], 'cp').iloc[:,0],utils.filter_data(df[dict_from_to[c][1]], 'lp').iloc[:,0]])
                else:
                    if c_split[dict_params['$j']] == dict_competitors['mapfre']:
                        # use mapfre
                        h_name = 'mapfre'
                    else:
                        # use mercado
                        h_name = 'mercado'

                    # get historic, cp and lp
                    v_last = v_hist[-1]
                    if h_last_var is None:  # use mapfre or mercado
                        c_h = c_split.copy()
                        c_h[dict_params['$j']] = dict_competitors[h_name]
                        if c_split[dict_params['$j']] == dict_premium['vida']:
                            c_h[dict_params['$r']] = dict_premium['vida']
                        else:
                            c_h[dict_params['$r']] = dict_premium['no vida']
                        c_h = '_'.join(c_h)
                        if (c_h not in df) or (c_h not in dict_from_to) or (dict_from_to[c_h][1] not in df):
                            continue
                    
                        # linear interpolation
                        h_cp_index = utils.filter_data(df[c_h], 'cp')[c_h].index
                        h_lp = utils.filter_data(df[dict_from_to[c_h][1]], 'lp')[dict_from_to[c_h][1]]
                        h_lp_index = h_lp.index
                        h_last = h_lp[-1]
                    elif isinstance(h_last_var, str):
                        if h_last_var == "ROE":
                            # get 10 year interest rate
                            c_h = utils.get_variable_name(
                                df_info_var.loc[3][dict_periods['lp']], 
                                scenario_code=scenario,
                                d={k:c_split[v] for k,v in dict_params.items()}
                            )
                            # if no data, use US
                            if c_h not in df:
                                c_h = utils.get_variable_name(
                                    df_info_var.loc[3][dict_periods['lp']], 
                                    country_code=dict_countries['estados unidos'],
                                    region_code=dict_region_n['estados unidos'],
                                    region_code_2=dict_region_g['estados unidos'],
                                    scenario_code=scenario,
                                )
                            # check other value in case US 10yrate is saved with another name
                            if c_h not in df:
                                c_h = utils.get_variable_name(
                                    df_info_var.loc[3][dict_periods['lp']], 
                                    country_code=dict_countries['estados unidos'],
                                    scenario_code=scenario,
                                    default='\\',
                                )
                            h_cp_index = utils.filter_data(df[[k for k in df.columns if 'cp' in k][0]], 'cp').index
                            # get last value of lp and add 6%
                            h_lp = utils.filter_data(df[c_h], 'lp').iloc[:,0]
                            h_lp_index = h_lp.index
                            h_last = (h_lp[-1] + 6)/100
                        else:
                            raise Exception("Unknown last variable")
                    else:
                        h_last = float(h_last_var)
                        h_lp_index = utils.filter_data(df[[k for k in df.columns if 'lp' in k][0]], 'lp').index
                        h_cp_index = utils.filter_data(df[[k for k in df.columns if 'cp' in k][0]], 'cp').index

                    v_fut = pd.Series(data = np.linspace(v_last,h_last,h_cp_index.shape[0] + h_lp_index.shape[0] + 1)[1:], index=h_cp_index.to_list() + h_lp_index.to_list())

                # combine hist, cp and lp and apply hpc filter
                v_all = pd.concat([v_hist,v_fut]).interpolate(method='linear', fill_value="extrapolate", limit_direction='both', limit_area=None)
                hp_cycle, hp_trend = hpfilter(v_all, lamb=self.HP_LAMBDA)

                # save for HP variable
                if hp_num is not None:
                    hp_names = [utils.get_variable_name(
                        p, 
                        scenario_code=scenario,
                        d={k:c_split[v] for k,v in dict_params.items()}
                    ) for p in v_periods_hp]
                    hp_hist = utils.filter_data(hp_trend, 'hist')
                    hp_hist.columns = [hp_names[0]]
                    hp_cp = utils.filter_data(hp_trend, 'cp')
                    hp_cp.columns = [hp_names[1]]
                    hp_lp = utils.filter_data(hp_trend, 'lp')
                    hp_lp.columns = [hp_names[2]]

                    data.extend([
                        hp_hist,
                        hp_cp,
                        hp_lp, 
                    ])

                # if variable cp and lp does not already exist
                if not exists and not simplified:
                    # separate cp and lp and save
                    v_cp = utils.filter_data(hp_trend, 'cp')
                    v_cp.columns = [dict_from_to[c][0]]
                    v_lp = utils.filter_data(hp_trend, 'lp')
                    v_lp.columns = [dict_from_to[c][1]]

                    data.extend([
                        v_cp, 
                        v_lp,
                    ])

                    # CALCULATE LEVEL VALUES

                    # get level names
                    dict_level = {k:c_split[v] for k,v in dict_params.items()}
                    level_periods = [utils.get_variable_name(k, scenario_code=scenario, d=dict_level) for k in v_periods_level]

                    # check if historical levels are in df
                    if level_periods[0] not in df:
                        continue

                    # calculate levels from rates
                    h_rates = pd.concat([v_cp[v_cp.columns[0]],v_lp[v_lp.columns[0]]])
                    h_hist_level = utils.filter_data(df[level_periods[0]], 'hist')[level_periods[0]]
                    h_fut_level = pd.Series(data=np.concatenate((h_hist_level.to_list()[-4:], np.zeros(h_rates.shape))), index=h_hist_level.index.to_list()[-4:] + h_rates.index.to_list())
                    
                    # level[t-4] * rate
                    for k in range(4, h_fut_level.shape[0]):
                        h_fut_level[k] = h_fut_level[-4 + k] * (1 + h_rates[k - 4])        

                    # separate cp and lp and save
                    h_fut_level = pd.DataFrame(h_fut_level.to_list(), index=h_fut_level.index)
                    v_cp_level = utils.filter_data(h_fut_level, 'cp')
                    v_cp_level.columns = [level_periods[1]]
                    v_lp_level = utils.filter_data(h_fut_level, 'lp')
                    v_lp_level.columns = [level_periods[2]]

                    data.extend([v_cp_level, v_lp_level])
        return data

    def _get_ratio_combinado(self, scenario:str):
        df_info_var = self.df_info_dict['variables']
        df = self.df
        dict_periods = utils.DATA_INFO_VARIABLES['VARIABLES_PERIODS']
        
        data = []
        for var_rate_num,var_level_num, hp_num, var_comb_nums in self.RC_VARS:
            # get level variable
            l = df_info_var.loc[var_level_num]
            l_hist_name = l[dict_periods['hist']]
            l_periods = [
                l[dict_periods[k]] for k in ['cp', 'lp']
            ]

            # get hp variable
            if hp_num is not None:
                v_hp = df_info_var.loc[hp_num]
                v_periods_hp = [
                    v_hp[dict_periods[k]] for k in ['hist', 'cp', 'lp']
                ]

            # get rate variable
            r = df_info_var.loc[var_rate_num]
            r_periods = [
                r[dict_periods[k]] for k in ['hist', 'cp', 'lp']
            ]

            # get combination levels
            combination_levels = [[df_info_var.loc[k][dict_periods[j]] for j in ['cp', 'lp']] for k in var_comb_nums]

            # dictionary of parameter positions
            dict_params = {k:idx for idx, k in enumerate(combination_levels[0][0].split('_')) if '$' in k}

            # columns from
            v_filter = utils.get_variable_name(
                combination_levels[0][0],
                scenario_code=scenario,
                default='.*'
            )
            pattern = re.compile(v_filter.replace('\\', '\\\\'))
            hist_cols = np.asarray([k for k in df.columns if pattern.match(k)])
            hist_cols_split = [k.split('_') for k in hist_cols]

            for c, c_split in zip(hist_cols, hist_cols_split):
                c_lp = utils.get_variable_name(combination_levels[0][1], scenario_code=scenario, d={k:c_split[v] for k,v in dict_params.items()})
                v_sum_cp = df[c]
                v_sum_lp = df[c_lp]
                try:
                    for k in combination_levels[1:]:
                        k_cp = utils.get_variable_name(k[0], scenario_code=scenario, d={k:c_split[v] for k,v in dict_params.items()})
                        k_lp = utils.get_variable_name(k[1], scenario_code=scenario, d={k:c_split[v] for k,v in dict_params.items()})
                        # sum cp if available
                        if k_cp in df:
                            v_sum_cp += df[k_cp]
                        # sum lp if available
                        if k_lp in df:
                            v_sum_lp += df[k_lp]
                except KeyError:
                    continue

                v_sum_cp = pd.DataFrame(v_sum_cp.to_list(), index=v_sum_cp.index, 
                    columns=[utils.get_variable_name(
                        l_periods[0], 
                        scenario_code=scenario,
                        d={k:c_split[v] for k,v in dict_params.items()}
                    )
                ])
                v_sum_cp = utils.filter_data(v_sum_cp, 'cp')
                v_sum_lp = pd.DataFrame(v_sum_lp.to_list(), index=v_sum_lp.index, 
                    columns=[utils.get_variable_name(
                        l_periods[1], 
                        scenario_code=scenario,
                        d={k:c_split[v] for k,v in dict_params.items()}
                    )
                ])
                v_sum_lp = utils.filter_data(v_sum_lp, 'lp')
                data.extend([v_sum_cp, v_sum_lp])

                # CALCULATE RATES
                l_hist_name = utils.get_variable_name(l_hist_name, scenario_code=scenario, d={k:c_split[v] for k,v in dict_params.items()})
                if l_hist_name not in df:
                    continue

                # get level data
                l_full = pd.concat([utils.filter_data(df[l_hist_name], 'hist')[l_hist_name][-4:], v_sum_cp.iloc[:,0], v_sum_lp.iloc[:,0]])
                # l_full = utils.filter_data(df[l_hist_name], 'hist')[l_hist_name][-4:].to_list() + v_sum_cp.iloc[:,0].to_list() + v_sum_lp.iloc[:,0].to_list()

                # calculate rates
                v_fut = l_full/l_full.shift(4) - 1
                # r_fut = []
                # for k in range(v_sum_cp.shape[0] + v_sum_lp.shape[0]):
                #     if l_full[k] == 0:
                #         r_fut.append(float("nan"))
                #     else:
                #         r_fut.append(
                #             (l_full[k+4] / l_full[k]) - 1
                #         )

                # v_fut = pd.Series(r_fut, index=v_sum_cp.index.to_list() + v_sum_lp.index.to_list())
                v_hist_name = utils.get_variable_name(r_periods[0], scenario_code=scenario, d={k:c_split[v] for k,v in dict_params.items()})
                
                # combine hist, cp and lp and apply hpc filter
                if hp_num is not None and v_hist_name in df:
                    v_all = pd.concat([
                        utils.filter_data(df[v_hist_name], 'hist').iloc[:,0],
                        utils.filter_data(v_fut, 'cp').iloc[:,0],
                        utils.filter_data(v_fut, 'lp').iloc[:,0],
                    ])
                    hp_cycle, hp_trend = hpfilter(v_all, lamb=self.HP_LAMBDA)

                    # save for HP variable
                    hp_names = [utils.get_variable_name(
                        p, 
                        scenario_code=scenario,
                        d={k:c_split[v] for k,v in dict_params.items()}
                    ) for p in v_periods_hp]
                    hp_hist = utils.filter_data(hp_trend, 'hist')
                    hp_hist.columns = [hp_names[0]]
                    hp_cp = utils.filter_data(hp_trend, 'cp')
                    hp_cp.columns = [hp_names[1]]
                    hp_lp = utils.filter_data(hp_trend, 'lp')
                    hp_lp.columns = [hp_names[2]]
                else:
                    hp_trend = v_fut

                # separate cp and lp and save
                v_cp = utils.filter_data(hp_trend, 'cp')
                v_cp.columns = [utils.get_variable_name(r_periods[1], scenario_code=scenario, d={k:c_split[v] for k,v in dict_params.items()})]
                v_lp = utils.filter_data(hp_trend, 'lp')
                v_lp.columns = [utils.get_variable_name(r_periods[2], scenario_code=scenario, d={k:c_split[v] for k,v in dict_params.items()})]

                data.extend([
                    hp_hist,
                    hp_cp,
                    hp_lp, 
                    v_cp, 
                    v_lp,
                ])
                
        return data
