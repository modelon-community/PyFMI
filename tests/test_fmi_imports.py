import pytest
import pyfmi.fmi as fmi

@pytest.mark.parametrize("x",
    [
        'CENTRAL_DIFFERENCE_EPS',
        'DeclaredType2',
        'EnumerationType2',
        'FMI1_DO_STEP_STATUS',
        'FMI1_LAST_SUCCESSFUL_TIME',
        'FMI1_PENDING_STATUS',
        'FMI2_BOOLEAN',
        'FMI2_CALCULATED_PARAMETER',
        'FMI2_CONSTANT',
        'FMI2_CONTINUOUS',
        'FMI2_DISCRETE',
        'FMI2_DO_STEP_STATUS',
        'FMI2_ENUMERATION',
        'FMI2_FALSE',
        'FMI2_FIXED',
        'FMI2_INDEPENDENT',
        'FMI2_INITIAL_APPROX',
        'FMI2_INITIAL_CALCULATED',
        'FMI2_INITIAL_EXACT',
        'FMI2_INITIAL_UNKNOWN',
        'FMI2_INPUT',
        'FMI2_INTEGER',
        'FMI2_KIND_CONSTANT',
        'FMI2_KIND_DEPENDENT',
        'FMI2_KIND_DISCRETE',
        'FMI2_KIND_FIXED',
        'FMI2_KIND_TUNABLE',
        'FMI2_LAST_SUCCESSFUL_TIME',
        'FMI2_LOCAL',
        'FMI2_OUTPUT',
        'FMI2_PARAMETER',
        'FMI2_PENDING_STATUS',
        'FMI2_REAL',
        'FMI2_STRING',
        'FMI2_TERMINATED',
        'FMI2_TRUE',
        'FMI2_TUNABLE',
        'FMI2_UNKNOWN',
        'FMI_ALIAS',
        'FMI_BOOLEAN',
        'FMI_CONSTANT',
        'FMI_CONTINUOUS',
        'FMI_CS_STANDALONE',
        'FMI_CS_TOOL',
        'FMI_DEFAULT_LOG_LEVEL',
        'FMI_DERIVATIVES',
        'FMI_DISCARD',
        'FMI_DISCRETE',
        'FMI_ENUMERATION',
        'FMI_ERROR',
        'FMI_FALSE',
        'FMI_FATAL',
        'FMI_INPUT',
        'FMI_INPUTS',
        'FMI_INTEGER',
        'FMI_INTERNAL',
        'FMI_ME',
        'FMI_NEGATED_ALIAS',
        'FMI_NONE',
        'FMI_NO_ALIAS',
        'FMI_OK',
        'FMI_OUTPUT',
        'FMI_OUTPUTS',
        'FMI_PARAMETER',
        'FMI_PENDING',
        'FMI_REAL',
        'FMI_REGISTER_GLOBALLY',
        'FMI_STATES',
        'FMI_STRING',
        'FMI_TRUE',
        'FMI_WARNING',
        'FMUException',
        'FMUModelBase',
        'FMUModelBase2',
        'FMUModelCS1',
        'FMUModelCS2',
        'FMUModelME1',
        'FMUModelME2',
        'FMUState2',
        'FORWARD_DIFFERENCE_EPS',
        'GLOBAL_FMU_OBJECT',
        'GLOBAL_LOG_LEVEL',
        'IntegerType2',
        'InvalidBinaryException',
        'InvalidFMUException',
        'InvalidOptionException',
        'InvalidVersionException',
        'InvalidXMLException',
        'LogHandler',
        'LogHandlerDefault',
        'ModelBase',
        'PyEventInfo',
        'RealType2',
        'ScalarVariable',
        'ScalarVariable2',
        'TimeLimitExceeded',
        'WorkerClass2',
        'check_fmu_args',
        'create_temp_dir',
        'load_fmu',
    ]
)
def test_import(x):
    """Temporary; to ensure we do not break any imports during re-factoring."""
    getattr(fmi, x)