# -*- coding: utf-8 -*-

#    Copyright (C) 2009 Modelon AB
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3 of the License.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module containing XML parser and validator providing an XML data structure based 
on the XML schemas fmiModelDescription.xsd, fmiExtendedModelDescription.xsd and 
jmodelicaModelDescription.xsd which can be used to extract information from an 
XML file - provided the XML file will validate with the above schemas.
"""
import os.path

from lxml import etree
import numpy as N

int = N.int32
N.int = N.int32
uint = N.uint32

# ==== Enumerations used in the module ==== #

# Alias data
NO_ALIAS = 0
ALIAS = 1
NEGATED_ALIAS = -1

# Variability
CONTINUOUS = 0
CONSTANT = 1
PARAMETER = 2
DISCRETE = 3

# Variable category
ALGEBRAIC = 0
STATE = 1
DEPENDENT_CONSTANT = 2
INDEPENDENT_CONSTANT = 3
DEPENDENT_PARAMETER = 4
INDEPENDENT_PARAMETER = 5
DERIVATIVE = 6

# causality
INTERNAL = 0
INPUT = 1
OUTPUT = 2
NONE = 3

# types
REAL = 0
INTEGER = 1
STRING = 2
BOOLEAN = 3
ENUMERATION = 4

#=======================================================================

# ==== Internal helper functions ====#

def _translate_xmlbool(xmlbool):
    """ 
    Helper function which translates strings 'true' and 'false' to bool types.
    """
    if xmlbool == 'false':
        return False
    elif xmlbool == 'true':
        return True
    else:
        raise Exception('The xml boolean '+str(xmlbool)+
            ' does not have a valid value.')
            
def _translate_variability(variability):
    """ 
    Helper function which translates strings from the attribute variability to 
    corresponding enumeration value.
    """
    if variability == "continuous":
        return CONTINUOUS
    elif variability == "constant":
        return CONSTANT
    elif variability == "parameter":
        return PARAMETER
    elif variability == "discrete":
        return DISCRETE
    else:
        raise XMLException("Variability: "+str(variability)+" is unknown.")

def _translate_alias(alias):
    """ 
    Helper function which translates strings from the attribute alias to 
    corresponding enumeration value.
    """
    if alias == "noAlias":
        return NO_ALIAS
    elif alias == "alias":
        return ALIAS
    elif alias == "negatedAlias":
        return NEGATED_ALIAS
    else:
        raise XMLException("Alias: "+ str(alias) + " is unknown.")
            
def _translate_variable_category(category):
    """ 
    Helper function which translates strings from the attribute variable 
    category to corresponding enumeration value.
    """
    if category == "algebraic":
        return ALGEBRAIC
    elif category == "state":
        return STATE
    elif category == "dependentConstant":
        return DEPENDENT_CONSTANT
    elif category == "independentConstant":
        return INDEPENDENT_CONSTANT
    elif category == "dependentParameter":
        return DEPENDENT_PARAMETER
    elif category == "independentParameter":
        return INDEPENDENT_PARAMETER
    elif category == "derivative":
        return DERIVATIVE
    else:
        raise XMLException("Variable category: "+
            str(category)+" is unknown.")
    
def _translate_causality(causality):
    """ 
    Helper function which translates strings from the attribute causality to 
    corresponding enumeration value.
    """
    if causality == "internal":
        return INTERNAL
    elif causality == "input":
        return INPUT
    elif causality == "output":
        return OUTPUT
    elif causality == "none":
        return NONE
    else:
        raise XMLException("Causality: "+str(causality)+" is unknown.")
        
def _translate_fundamental_type(type):
    """ 
    Helper function which translates strings from the scalar variable 
    fundamental type to corresponding enumeration value.
    """
    if isinstance(type, Real):
        return REAL
    elif isinstance(type, Integer):
        return INTEGER
    elif isinstance(type, String):
        return STRING
    elif isinstance(type, Boolean):
        return BOOLEAN
    elif isinstance(type, Enumeration):
        return ENUMERATION
    else:
        raise XMLException("Unknown type for variable: "+ 
            str(type))

def _parse_XML(filename, schemaname=''):
    """ 
    Parse and validate (optional) an XML file.

    Parse an XML file and return an object representing the parsed XML. If the 
    optional parameter schemaname is set the XML file is also validated against 
    the XML Schema file provided before parsing. 

    Parameters::
    
        filename -- 
            Name of XML file to parse including absolute or relative path.
            
        schemaname --
            Name of XML Schema file including absolute or relative path.
            
            Default: Empty string.
    
    Exceptions::
       
        XMLException -- 
            If the XML file can not be read or is not well-formed. If a schema 
            is present and if the schema file can not be read, is not 
            well-formed or if the validation fails. 
    
    Returns::
       
        A reference to the ElementTree object containing the parsed XML.
    """

    try:
        element_tree = etree.ElementTree(file=filename)
    except etree.XMLSyntaxError, detail:
        raise XMLException("The XML file: %s is not well-formed. %s" 
            %(filename, detail))

    if schemaname:
        try:
            schemadoc = etree.ElementTree(file=schemaname)
        except etree.XMLSyntaxError, detail:
            raise XMLException("The XMLSchema: %s is not well-formed. %s" 
                %(schemaname, detail))         
        
        schema = etree.XMLSchema(schemadoc)

        result = schema.validate(xmldoc)
    
        if not result:
            raise XMLException("The XML file: %s is not valid \
                according to the XMLSchema: %s." 
                %(filename, schemaname))
    
    return element_tree

class ModelDescription:

    def __init__(self, filename, schemaname=''):
        """ 
        Create an XML document object representation.
        
        Parse an XML document and create a full XML document object 
        representation. Validate against XML schema before parsing if the 
        parameter schemaname is set.
        
        Parameters::
        
            filename --
                The name of the XML file to parse.
                
            schemaname --
                The name of the XSD file to validate against.
                Default: Empty string (no validation).
        """
        # set up cache, parse XML file and obtain the root
        self.function_cache = XMLFunctionCache()
        element_tree = _parse_XML(filename, schemaname)
        root = element_tree.getroot()
        
        # populate vref tuple already here in init (done during 
        # _parse_element_tree) since this tuple will be used in almost 
        # all "complex methods"
        self._vrefs = []
        self._vrefs_noAlias = []
        
        # build internal data structure from XML file
        self._parse_element_tree(root)
        
        # create tuple
        self._vrefs = tuple(self._vrefs)
        self._vrefs_noAlias = tuple(self._vrefs_noAlias)
        
        self._nu = 0
        self._ny = 0
        self._ncu = 0
        self._ncy = 0

        for v in self.get_model_variables():
            if v.get_causality()==INPUT:
                self._nu = self._nu + 1
                if v.get_variability()==CONTINUOUS:
                    self._ncu = self._ncu + 1
            if v.get_causality()==OUTPUT:
                self._ny = self._ny + 1
                if v.get_variability()==CONTINUOUS:
                    self._ncy = self._ncy + 1


    def _parse_element_tree(self, root):
        """ 
        Parse the XML element tree and build up internal data structure. 
        """
        # model (root) attributes
        self._fill_attributes(root)
            
        # unit definitions
        self._fill_unit_definitions(root)
        
        # type definitions
        self._fill_type_definitions(root)
        
        # default experiment
        self._fill_default_experiment(root)
        
        # vendor annotations
        self._fill_vendor_annotations(root)
        
        # model variables
        self._fill_model_variables(root)
        
        # fill optimization
        self._fill_optimization(root)
              
    def _fill_attributes(self, root):
        """ 
        Set the Model Description attributes. 
        """
        # declare attributes with default values
        self._attributes = {'fmiVersion':'',
                           'modelName':'',
                           'modelIdentifier':'',
                           'guid':'',
                           'description':'',
                           'author':'',
                           'version':'',
                           'generationTool':'',
                           'generationDateAndTime':'',
                           'variableNamingConvention':'flat',
                           'numberOfContinuousStates':'',
                           'numberOfEventIndicators':''}
                           
        # update attribute dict with attributes from XML file
        self._attributes.update(root.attrib) 
            
    def _fill_unit_definitions(self, root):
        """ 
        Create the unit definitions data structure and fill with data from the 
        XML file.
        """
        self._unit_definitions = []
        
        e_unitdefs = root.find('UnitDefinitions')
        if e_unitdefs != None:
            # list of base units (xml elements)
            e_baseunits = e_unitdefs.getchildren()
            for e_baseunit in e_baseunits:
                self._unit_definitions.append(BaseUnit(e_baseunit))
                
    def _fill_type_definitions(self, root):
        """ 
        Create the type definitions data structure and fill with data from the 
        XML file.
        """
        self._type_definitions = []
        
        e_typedefs = root.find('TypeDefinitions')
        if e_typedefs != None:
            # list of types
            e_types = e_typedefs.getchildren()
            for e_type in e_types:
                self._type_definitions.append(Type(e_type))
                
    def _fill_default_experiment(self, root):
        """ 
        Create the default experiment data structure and fill with data from the 
        XML file.
        """
        self._default_experiment = None
        
        e_defaultexperiment = root.find('DefaultExperiment')
        if e_defaultexperiment != None:
            self._default_experiment = DefaultExperiment(e_defaultexperiment)
    
    def _fill_vendor_annotations(self, root):
        """ 
        Create the vendor annotations data structure and fill with data from the 
        XML file.
        """
        self._vendor_annotations = []
        
        e_vendorannotations = root.find('VendorAnnotations')
        if e_vendorannotations != None:
            # list of tools
            e_tools = e_vendorannotations.getchildren()
            for e_tool in e_tools:
                self._vendor_annotations.append(Tool(e_tool))
                
    def _fill_model_variables(self, root):
        """ 
        Create the model variables data structure (list with all scalar 
        variables) and fill with data from the XML file.
        """
        self._model_variables = []
        self._model_variables_dict = {}
        
        e_modelvariables = root.find('ModelVariables')
        if e_modelvariables != None:
            # list of scalar variables
            e_scalarvariables = e_modelvariables.getchildren()
            for e_scalarvariable in e_scalarvariables:
                sv = ScalarVariable(e_scalarvariable)
                self._model_variables.append(sv)
                
                # fill model variables dicts
                self._model_variables_dict[sv.get_name()] = sv
                
                # fill vref and vref no alias lists
                self._vrefs.append(sv.get_value_reference())
                if sv.get_alias() == NO_ALIAS:
                    self._vrefs_noAlias.append(sv.get_value_reference())
                
    def _fill_optimization(self, root):
        """ 
        Create the optimization data structure (if any) and fill with data from 
        the XML file.
        """
        self._optimization = None
        
        try:
            opt=root.nsmap['opt']
        except KeyError:
            # no optimization part in xml
            return
            
        ns="{"+opt+"}"
        e_optimization = root.find(ns+'Optimization')
        if e_optimization != None:
            self._optimization = Optimization(e_optimization)
            
    def get_fmi_version(self):
        """ 
        Get model attribute fmi version.
        
        Returns::
        
            The FMI version attribute as string.
        """
        return self._attributes['fmiVersion']
        
    def get_model_name(self):
        """ 
        Get model attribute name.
        
        Returns::
        
            The model name attribute value as string.
        """
        return self._attributes['modelName']
        
    def get_model_identifier(self):
        """ 
        Get model attribute model identifier.
        
        Returns::
        
            The model identifier attribute value as string.
        """
        return self._attributes['modelIdentifier']
        
    def get_guid(self):
        """ 
        Get model attribute GUID.
        
        Returns::
        
            The GUID attribute value as string.
        """
        return self._attributes['guid']
        
    def get_description(self):
        """ 
        Get model attribute description.
        
        Returns::
        
            The description attribute value as string (empty string if not 
            specified in XML).
        """
        return self._attributes['description']
    
    def get_author(self):
        """ 
        Get model attribute author.
        
        Returns::
        
            The author attribute value as string (empty string if not specified 
            in XML).
        """
        return self._attributes['author']
        
    def get_version(self):
        """ 
        Get model attribute version (of FMU). 
        
        Returns::
        
            The version attribute value as float if set, otherwise None.
        """
        if self._attributes['version'] == '':
            return None
        return float(self._attributes['version'])
        
    def get_generation_tool(self):
        """ 
        Get model attribute generation tool.
        
        Returns::
        
            The generation tool attribute value as string (empty string if not 
            specified in XML).
        """
        return self._attributes['generationTool']
        
    def get_generation_date_and_time(self):
        """ 
        Get model attribute date and time.
        
        Returns::
        
            The date and time attribute value as string (empty string if not 
            specified in XML).
        """
        return self._attributes['generationDateAndTime']
        
    def get_variable_naming_convention(self):
        """ 
        Get model attribute variable naming convention.
        
        Returns::
        
            The variable naming convention attribute value as string.
        """
        return self._attributes['variableNamingConvention']
        
    def get_number_of_continuous_states(self):
        """ 
        Get model attribute number of continuous states.
        
        Returns::
        
            The number of continuous states attribute value as unsigned int.
         """
        if self._attributes['numberOfContinuousStates'] == '':
            return None
        return uint(self._attributes['numberOfContinuousStates'])
        
    def get_number_of_event_indicators(self):
        """ 
        Get model attribute number of event indicators.
        
        Returns::
        
            The number of event indicators attribute value as unsigned int.
         """
        if self._attributes['numberOfEventIndicators'] == '':
            return None
        return uint(self._attributes['numberOfEventIndicators'])
                    
    def get_number_of_inputs(self):
        """
        Get the number of inputs.
        
        Returns::
        
            The number of inputs.
        """
        return self._nu
    
    def get_number_of_continuous_inputs(self):
        """
        Get the number of continuous inputs.
        
        Returns::
        
            The number of continuous inputs.
        """
        return self._ncu
    
    def get_number_of_outputs(self):
        """
        Get the number of outputs.
        
        Returns::
        
            The number of outputs.
        """
        return self._ny
    
    def get_number_of_continuous_outputs(self):
        """
        Get the number of continuous outputs.
        
        Returns::
        
            The number of continuous outputs.
        """
        return self._ncy
            
    def get_continous_outputs_value_references(self):
        """
        Get the value references of the of continuous outputs.
        
        Returns::
        
            A list of value references.
        """
        yc_vrefs=[]
        for v in self.get_model_variables():
            if v.get_causality()==OUTPUT:
                if v.get_variability()==CONTINUOUS:
                    yc_vrefs.append(v.get_value_reference())
        return sorted(yc_vrefs)

    def get_continous_inputs_value_references(self):
        """
        Get the value references of the of continuous inputs.
        
        Returns::
        
            A list of value references.
        """
        uc_vrefs=[]
        for v in self.get_model_variables():
            if v.get_causality()==INPUT:
                if v.get_variability()==CONTINUOUS:
                    uc_vrefs.append(v.get_value_reference())
        return sorted(uc_vrefs)
        
    def get_unit_definitions(self):
        """ 
        Get all unit definitions set in model.
        
        Returns::
        
            A list of unit definitions (type: BaseUnit)
        """
        return self._unit_definitions
        
    def get_type_definitions(self):
        """ 
        Get all type definitions set in model.
        
        Returns::
        
            A list of type definitions (type: Type)
        """
        return self._type_definitions
        
    def get_default_experiment(self):
        """ 
        Get default experiment data set in model.
        
        Returns::
        
            An object of type DefaultExperiment or None if not set.
        """
        return self._default_experiment
        
    def get_vendor_annotations(self):
        """ 
        Get all vendor annotations set in model.
        
        Returns::
        
            A list of vendor annotations (type: Tool)
        """
        return self._vendor_annotations
            
    def get_model_variables(self):
        """ 
        Get all variables in model. 
        
        Returns::
        
            A list of all variables (type: ScalarVariable)
        """
        return self._model_variables
        
    def get_optimization(self):
        """ 
        Get optimization data set in model.
        
        Returns::
        
            An object of type Optimization if model is Optimica, otherwise None.
        """
        return self._optimization
        
    # ========== Here begins the more complex functions ================
    
    def get_value_reference(self, variablename, ignore_cache=False):
        """ 
        Get the value reference given a variable name.
        
        Parameters::
        
            variablename -- 
                The name of the variable.
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
            
        Returns::
        
            The value reference for the variable passed as argument.
        
        Raises::
        
            XMLEexception if variable was not found.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_value_reference', 
                variablename)
                
        sv = self._model_variables_dict.get(variablename)
        if sv != None:
            return sv.get_value_reference()
        else:
            raise XMLException("Variable: "+str(variablename)+" was not found \
            in the model.")
        
    def is_alias(self, variablename, ignore_cache=False):
        """ 
        Find out if variable is an alias or negated alias.
        
        Parameters::
        
            variablename --
                The name of the variable.
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            True if variable is alias or negated alias, False otherwise.
        
        Raises::
        
            XMLException if variable was not found.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'is_alias', variablename)
                
        sv = self._model_variables_dict.get(variablename)
        if sv != None:
            return (sv.get_alias() != NO_ALIAS)
        else:
            raise XMLException("Variable: "+str(variablename)+" was not found \
            in model.")

    def is_negated_alias(self, variablename, ignore_cache=False):
        """ 
        Find out if variable is a negated alias or not.
        
        Parameters::
        
            variablename --
                The name of the variable.
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            True if variable is a negated alias, False otherwise.
        
        Raises::
        
            XMLException if variable was not found.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'is_negated_alias', 
            variablename)
                
        sv = self._model_variables_dict.get(variablename)
        if sv != None:
            return (sv.get_alias() == NEGATED_ALIAS)
        else:
            raise XMLException("Variable: "+str(variablename)+" was not found \
            in model.")
        
    def is_constant(self, variablename, ignore_cache=False):
        """ 
        Find out if variable is a constant or not. 
        
        Parameters::
        
            variablename --
                The name of the variable.
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            True if variable is a constant, False otherwise.
        
        Raises::
        
            XMLException if variable was not found.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'is_constant', variablename)
                
        sv = self._model_variables_dict.get(variablename)
        if sv != None:
            return sv.get_variability() == CONSTANT
        else:
            raise XMLException("Variable: "+str(variablename)+" was not found \
            in model.")
        
    def get_data_type(self, variablename, ignore_cache=False):
        """ 
        Get data type of variable. 
        
        Parameters::
        
            variablename --
                The name of the variable.
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            The type of the variable, REAL, INTEGER, BOOLEAN or STRING.
        
        Raises::
        
            XMLException if variable was not found.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_data_type', variablename)
                
        sv = self._model_variables_dict.get(variablename)
        if sv != None:
            return _translate_fundamental_type(sv.get_fundamental_type())
        else:
            raise XMLException("Variable: "+str(variablename)+" was not found \
            in model.")

    def get_aliases_for_variable(self, variablename, ignore_cache=False):
        """ 
        Get a list of all alias variables belonging to the aliased variable 
        passed as argument to the function along with a list of booleans 
        indicating whether the alias variable should be negated or not.
        
        Parameters::
            
            variable --
                The aliased variable.
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
            
        Returns::
                
                A tuple of lists, the first containing the names of the alias 
                variables, the second containing booleans for each alias 
                variable indicating whether the variable is a negated variable 
                or not.
                
                A tuple of empty lists if the variable has no alias variables. 
                
                None if variable cannot be found in model.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_aliases_for_variable', 
                variablename)
                
        aliasnames = []
        isnegated = []
        
        variable = self._model_variables_dict.get(variablename)
        if variable != None:
            for sv in self.get_model_variables():
                if sv.get_value_reference() == variable.get_value_reference() and \
                    sv.get_name()!=variablename:
                        aliasnames.append(sv.get_name())
                        isnegated.append(sv.get_alias()== NEGATED_ALIAS)
            return aliasnames, isnegated
        return None

    def get_variability(self, variablename, ignore_cache=False):
        """ 
        Get variability of variable. 
        
        Parameters::
        
            variablename --
                The name of the variable.
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The variability of the variable, CONTINUOUS, CONSTANT, PARAMETER or 
            DISCRETE
        
        Raises::
        
            XMLException if variable was not found.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_variability', 
            variablename)
                
        sv = self._model_variables_dict.get(variablename)
        if sv != None:
            return sv.get_variability()
        else:
            raise XMLException("Variable: "+str(variablename)+" was not found \
            in model.")

    def get_variable_names(self, include_alias=True, ignore_cache=False):
        """ 
        Extract the names of the variables in a model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False

        Returns::
        
            A list of tuples containing value references and names respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self,'get_variable_names', 
                include_alias)

        names = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                names.append(sv.get_name())
            return zip(tuple(self._vrefs),tuple(names))

        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS:
                names.append(sv.get_name())
        return zip(tuple(self._vrefs_noAlias),tuple(names))
        
    def get_variable_aliases(self, ignore_cache=False):
        """ 
        Extract the alias data for each variable in the model.
        
        Parameters::
        
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value references and alias data 
            respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self,'get_variable_aliases')
        
        alias_data = []
        scalarvariables = self.get_model_variables()
        for sv in scalarvariables:
            alias_data.append(sv.get_alias())
        return zip(tuple(self._vrefs),tuple(alias_data))

    def get_variable_descriptions(self, include_alias=True, ignore_cache=False):
        """ 
        Extract the descriptions of the variables in a model.

        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False

        Returns::
        
            A list of tuples containing value reference and description 
            respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_variable_descriptions', 
                include_alias)
            
        descriptions = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                descriptions.append(sv.get_description())
            return zip(tuple(self._vrefs),tuple(descriptions))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS:
                descriptions.append(sv.get_description())
        return zip(tuple(self._vrefs_noAlias),tuple(descriptions))
        
    def get_variable_variabilities(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get the variability of the variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            A list of tuples containing value reference and variability 
            respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_variable_variabilities', 
                include_alias)
        
        variabilities = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                variabilities.append(sv.get_variability())
            return zip(tuple(self._vrefs),tuple(variabilities))
        
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS:
                variabilities.append(sv.get_variability())
        return zip(tuple(self._vrefs_noAlias),tuple(variabilities))

    def get_variable_nominal_attributes(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get the nominal attribute of the variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of nominal 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 
                'get_variable_nominal_attributes', include_alias)
        
        nominals = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if isinstance(ftype, Real):
                    nominals.append(ftype.get_nominal())
                else:
                    nominals.append(None)
            return zip(tuple(self._vrefs),tuple(nominals))
        
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if sv.get_alias() == NO_ALIAS:
                if isinstance(ftype, Real):
                    nominals.append(ftype.get_nominal())
                else:
                    nominals.append(None)
        return zip(tuple(self._vrefs_noAlias),tuple(nominals))

    def get_variable_fixed_attributes(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get the fixed attribute of the variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of fixed 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 
                'get_variable_fixed_attributes', include_alias)
        
        fixeds = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if isinstance(ftype, Real):
                    fixeds.append(ftype.get_fixed())
                else:
                    fixeds.append(None)
            return zip(tuple(self._vrefs),tuple(fixeds))
        
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if sv.get_alias() == NO_ALIAS:
                if isinstance(ftype, Real):
                    fixeds.append(ftype.get_fixed())
                else:
                    fixeds.append(None)
        return zip(tuple(self._vrefs_noAlias),tuple(fixeds))



    def get_variable_start_attributes(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get the start attributes of the variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of start 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 
                'get_variable_start_attributes', include_alias)
        
        start_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                start_attributes.append(sv.get_fundamental_type().get_start())
            return zip(tuple(self._vrefs), tuple(start_attributes))
       
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS:
                start_attributes.append(sv.get_fundamental_type().get_start())
        return zip(tuple(self._vrefs_noAlias), tuple(start_attributes))
        
    def get_all_real_variables(self, include_alias=True, ignore_cache=False):
        """ 
        Get all real variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of all ScalarVariables of type Real.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_all_real_variables', 
                include_alias)
                
        return self._get_all_variables(Real, include_alias)
        
    def get_all_string_variables(self, include_alias=True, ignore_cache=False):
        """ 
        Get all string variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of all ScalarVariables of type String.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_all_string_variables', 
                include_alias)
                
        return self._get_all_variables(String, include_alias)
        
    def get_all_integer_variables(self, include_alias=True, ignore_cache=False):
        """ 
        Get all integer variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of all ScalarVariables of type Integer.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_all_integer_variables', 
                include_alias)
                
        return self._get_all_variables(Integer, include_alias)

    def get_all_boolean_variables(self, include_alias=True, ignore_cache=False):
        """ 
        Get all boolean variables in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of all ScalarVariables of type Boolean.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_all_boolean_variables', 
                include_alias)
                
        return self._get_all_variables(Boolean, include_alias)

        
    def _get_all_variables(self, type, include_alias):
        """ 
        Helper function which returns all variables of type: 'type'.
        """
        typevars = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if isinstance(sv.get_fundamental_type(), type):
                    typevars.append(sv)
            return typevars
            
        for sv in scalarvariables:
            if isinstance(sv.get_fundamental_type(), type) and \
                sv.get_alias() == NO_ALIAS:
                    typevars.append(sv)
        return typevars

    def get_p_opt_variable_names(self, include_alias=True, ignore_cache=False):
        """ 
        Get the names of all optimized independent parameters.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and name respectively.
            
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_variable_names', 
                include_alias)
        
        vrefs = []
        names = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variability() == PARAMETER and \
                    sv.get_fundamental_type().get_free() == True:
                        vrefs.append(sv.get_value_reference())
                        names.append(sv.get_name())
            return zip(tuple(vrefs), tuple(names))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_variability() == PARAMETER and \
                sv.get_fundamental_type().get_free() == True:
                        vrefs.append(sv.get_value_reference())
                        names.append(sv.get_name())
        return zip(tuple(vrefs), tuple(names))
                
    def get_dx_variable_names(self, include_alias=True, ignore_cache=False):
        """ 
        Get the names of all derivatives.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and name respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_dx_variable_names', 
                include_alias)
        
        vrefs = []
        names = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
            return zip(tuple(vrefs), tuple(names))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
        return zip(tuple(vrefs), tuple(names))
                    
    def get_x_variable_names(self, include_alias=True, ignore_cache=False):
        """ 
        Get the names of all states.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
                
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and name respectively.
            
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_variable_names', 
                include_alias)
        
        vrefs = []
        names = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
            return zip(tuple(vrefs), tuple(names))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
        return zip(tuple(vrefs), tuple(names))
                    
    def get_u_variable_names(self, include_alias=True, ignore_cache=False):
        """ 
        Get the names of all inputs.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and name respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_variable_names', 
                include_alias)
        
        vrefs = []
        names = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
            return zip(tuple(vrefs), tuple(names))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() == INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
        return zip(tuple(vrefs), tuple(names))

    def get_w_variable_names(self, include_alias=True, ignore_cache=False):
        """ 
        Get the names of all algebraic variables.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and name respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_variable_names', 
                include_alias)
        
        vrefs = []
        names = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
            return zip(tuple(vrefs), tuple(names))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() != INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    names.append(sv.get_name())
        return zip(tuple(vrefs), tuple(names))

    def get_p_opt_start(self, include_alias=True, ignore_cache=False):
        """ 
        Get the start attributes of the independent paramenters 
        (variability:parameter, free: true) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of start 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_start', 
                include_alias)
        
        vrefs = []
        start_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variability() == PARAMETER and \
                    sv.get_fundamental_type().get_free() == True:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
            return zip(tuple(vrefs), tuple(start_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_variability() == PARAMETER and \
                sv.get_fundamental_type().get_free() == True:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
        return zip(tuple(vrefs), tuple(start_attributes))
                
    def get_dx_start(self, include_alias=True, ignore_cache=False):
        """ 
        Get the start attributes of the derivatives 
        (variable_category:derivative) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of start 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_dx_start', include_alias)
        
        vrefs = []
        start_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
            return zip(tuple(vrefs), tuple(start_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
        return zip(tuple(vrefs), tuple(start_attributes))
                    
    def get_x_start(self, include_alias=True, ignore_cache=False):
        """ 
        Get the start attributes of the states (variable_category:state) in the 
        model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of start 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_start', include_alias)
        
        vrefs = []
        start_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
            return zip(tuple(vrefs), tuple(start_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
        return zip(tuple(vrefs), tuple(start_attributes))
                    
    def get_u_start(self, include_alias=True, ignore_cache=False):
        """ 
        Get the start attributes of the inputs (variable_category:algebraic, 
        causality: input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of start 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_start', include_alias)
        
        vrefs = []
        start_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
            return zip(tuple(vrefs), tuple(start_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() == INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
        return zip(tuple(vrefs), tuple(start_attributes))

    def get_w_start(self, include_alias=True, ignore_cache=False):
        """ 
        Get the start attributes of the algebraic variables 
        (variable_category:algebraic, causality: not input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of start 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_start', include_alias)
        
        vrefs = []
        start_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
            return zip(tuple(vrefs), tuple(start_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() != INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    start_attributes.append(sv.get_fundamental_type().get_start())
        return zip(tuple(vrefs), tuple(start_attributes))

############################
    def get_dx_fixed(self, include_alias=True, ignore_cache=False):
        """ 
        Get the fixed attributes of the derivatives 
        (variable_category:derivative) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of fixed 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_dx_fixed', include_alias)
        
        vrefs = []
        fixed_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
            return zip(tuple(vrefs), tuple(fixed_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
        return zip(tuple(vrefs), tuple(fixed_attributes))
                    
    def get_x_fixed(self, include_alias=True, ignore_cache=False):
        """ 
        Get the fixed attributes of the states (variable_category:state) in the 
        model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of fixed 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_fixed', include_alias)
        
        vrefs = []
        fixed_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
            return zip(tuple(vrefs), tuple(fixed_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
        return zip(tuple(vrefs), tuple(fixed_attributes))
                    
    def get_u_fixed(self, include_alias=True, ignore_cache=False):
        """ 
        Get the fixed attributes of the inputs (variable_category:algebraic, 
        causality: input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of fixed 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_fixed', include_alias)
        
        vrefs = []
        fixed_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
            return zip(tuple(vrefs), tuple(fixed_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() == INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
        return zip(tuple(vrefs), tuple(fixed_attributes))

    def get_w_fixed(self, include_alias=True, ignore_cache=False):
        """ 
        Get the fixed attributes of the algebraic variables 
        (variable_category:algebraic, causality: not input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of fixed 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_fixed', include_alias)
        
        vrefs = []
        fixed_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
            return zip(tuple(vrefs), tuple(fixed_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() != INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    fixed_attributes.append(sv.get_fundamental_type().get_fixed())
        return zip(tuple(vrefs), tuple(fixed_attributes))

    def get_x_nominal(self, include_alias=True, ignore_cache=False):
        """ 
        Get the nominal attributes of the states (variable_category:state) in the 
        model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of nominal 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_nominal', include_alias)
        
        vrefs = []
        nominal_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    nominal_attributes.append(sv.get_fundamental_type().get_nominal())
            return zip(tuple(vrefs), tuple(nominal_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    nominal_attributes.append(sv.get_fundamental_type().get_nominal())
        return zip(tuple(vrefs), tuple(nominal_attributes))
                    
    def get_u_nominal(self, include_alias=True, ignore_cache=False):
        """ 
        Get the nominal attributes of the inputs (variable_category:algebraic, 
        causality: input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of nominal 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_nominal', include_alias)
        
        vrefs = []
        nominal_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    nominal_attributes.append(sv.get_fundamental_type().get_nominal())
            return zip(tuple(vrefs), tuple(nominal_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() == INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    nominal_attributes.append(sv.get_fundamental_type().get_nominal())
        return zip(tuple(vrefs), tuple(nominal_attributes))

    def get_w_nominal(self, include_alias=True, ignore_cache=False):
        """ 
        Get the nominal attributes of the algebraic variables 
        (variable_category:algebraic, causality: not input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of nominal 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_nominal', include_alias)
        
        vrefs = []
        nominal_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    nominal_attributes.append(sv.get_fundamental_type().get_nominal())
            return zip(tuple(vrefs), tuple(nominal_attributes))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
               sv.get_causality() != INPUT and \
               sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    nominal_attributes.append(sv.get_fundamental_type().get_nominal())
        return zip(tuple(vrefs), tuple(nominal_attributes))

    def get_p_opt_nominal(self, include_alias=True, ignore_cache=False):
        """ 
        Get the nominal attributes for all optimized parameters
        (variability:parameter, free: true).
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in
                the result. If False, only non-alias variables will be included
                in the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of nominal 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_nominal',
                                           include_alias)
        
        vrefs = []
        nominal_attributes = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if (sv.get_variability() == PARAMETER and 
                    ftype.get_free() == True):
                    vrefs.append(sv.get_value_reference())
                    nominal_attributes.append(ftype.get_nominal())
            return zip(tuple(vrefs), tuple(nominal_attributes))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if (sv.get_alias() == NO_ALIAS and 
                sv.get_variability() == PARAMETER and 
                ftype.get_free() == True):
                vrefs.append(sv.get_value_reference())
                nominal_attributes.append(ftype.get_nominal())
        return zip(tuple(vrefs), tuple(nominal_attributes))

#############################
    def get_p_opt_initial_guess(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and initial guess attribute for all optimized 
        independent parameters (variability:parameter, free: true).
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of initial 
            guess attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_initial_guess', 
                include_alias)
        vrefs = []
        initial_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Enumeration) and \
                    sv.get_variability() == PARAMETER and \
                    ftype.get_free() == True:
                        
                    vrefs.append(sv.get_value_reference())
                    initial_values.append(ftype.get_initial_guess())
            return zip(tuple(vrefs), tuple(initial_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Enumeration) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variability() == PARAMETER and \
                ftype.get_free() == True:
                    
                    vrefs.append(sv.get_value_reference())
                    initial_values.append(ftype.get_initial_guess())
        return zip(tuple(vrefs), tuple(initial_values))

    def get_dx_initial_guess(self, include_alias=True, ignore_cache=False):
        """ 
        Get the initial guess attribute of the derivatives 
        (variable_category:derivative) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of initial 
            guess attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_dx_initial_guess', 
                include_alias)
        
        vrefs = []
        initial_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Enumeration) and \
                    sv.get_variable_category() == DERIVATIVE:
                        
                    vrefs.append(sv.get_value_reference())
                    initial_values.append(ftype.get_initial_guess())
            return zip(tuple(vrefs), tuple(initial_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Enumeration) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == DERIVATIVE:
                   
                    vrefs.append(sv.get_value_reference())
                    initial_values.append(ftype.get_initial_guess())
        return zip(tuple(vrefs), tuple(initial_values))

    def get_x_initial_guess(self, include_alias=True, ignore_cache=False):
        """ 
        Get the initial guess attributes of the states (variable_category:state) 
        in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of initial 
            guess attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_initial_guess', 
                include_alias)
        
        vrefs = []
        initial_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Enumeration) and \
                    sv.get_variable_category() == STATE:
                    
                        vrefs.append(sv.get_value_reference())
                        initial_values.append(ftype.get_initial_guess())
            return zip(tuple(vrefs), tuple(initial_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Enumeration) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == STATE:
                    
                    vrefs.append(sv.get_value_reference())
                    initial_values.append(ftype.get_initial_guess())
        return zip(tuple(vrefs), tuple(initial_values))

    def get_u_initial_guess(self, include_alias=True, ignore_cache=False):
        """ 
        Get the initial guess attributes of the inputs 
        (variable_category:algebraic, causality: input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of initial 
            guess attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_initial_guess', 
                include_alias)
        
        vrefs = []
        initial_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Enumeration) and \
                    sv.get_causality() == INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                    
                        vrefs.append(sv.get_value_reference())
                        initial_values.append(ftype.get_initial_guess())
            return zip(tuple(vrefs), tuple(initial_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Enumeration) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    
                    vrefs.append(sv.get_value_reference())
                    initial_values.append(ftype.get_initial_guess())
        return zip(tuple(vrefs), tuple(initial_values))

    def get_w_initial_guess(self, include_alias=True, ignore_cache=False):
        """ 
        Get the initial guess attributes of the algebraic variables 
        (variable_category:algebraic, causality: not input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of initial 
            guess attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_initial_guess', 
                include_alias)
        
        vrefs = []
        initial_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Enumeration) and \
                    sv.get_causality() != INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                        
                        vrefs.append(sv.get_value_reference())
                        initial_values.append(ftype.get_initial_guess())
            return zip(tuple(vrefs), tuple(initial_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Enumeration) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    
                    vrefs.append(sv.get_value_reference())
                    initial_values.append(ftype.get_initial_guess())
        return zip(tuple(vrefs), tuple(initial_values))

    def get_p_opt_min(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and min attribute for all optimized independent 
        parameters (variability:parameter, free: true).
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of min 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_min', include_alias)
        vrefs = []
        min_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_variability() == PARAMETER and \
                    ftype.get_free() == True:
                        
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
            return zip(tuple(vrefs), tuple(min_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variability() == PARAMETER and \
                ftype.get_free() == True:
                    
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
        return zip(tuple(vrefs), tuple(min_values))

    def get_dx_min(self, include_alias=True, ignore_cache=False):
        """ 
        Get the min attribute of the derivatives (variable_category:derivative) 
        in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of min 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_dx_min', include_alias)
        
        vrefs = []
        min_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_variable_category() == DERIVATIVE:
                        
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
            return zip(tuple(vrefs), tuple(min_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == DERIVATIVE:
                   
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
        return zip(tuple(vrefs), tuple(min_values))

    def get_x_min(self, include_alias=True, ignore_cache=False):
        """ 
        Get the min attributes of the states (variable_category:state) in the 
        model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of min 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_min', include_alias)
        
        vrefs = []
        min_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_variable_category() == STATE:
                        
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
            return zip(tuple(vrefs), tuple(min_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == STATE:
                    
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
        return zip(tuple(vrefs), tuple(min_values))

    def get_u_min(self, include_alias=True, ignore_cache=False):
        """ 
        Get the min attributes of the inputs (variable_category:algebraic, 
        causality: input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of min 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_min', include_alias)
        
        vrefs = []
        min_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_causality() == INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                        
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
            return zip(tuple(vrefs), tuple(min_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
        return zip(tuple(vrefs), tuple(min_values))

    def get_w_min(self, include_alias=True, ignore_cache=False):
        """ 
        Get the min attributes of the algebraic variables 
        (variable_category:algebraic, causality: not input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of min 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_min', include_alias)
        
        vrefs = []
        min_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_causality() != INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                        
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
            return zip(tuple(vrefs), tuple(min_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    
                    vrefs.append(sv.get_value_reference())
                    min_values.append(ftype.get_min())
        return zip(tuple(vrefs), tuple(min_values))

    def get_p_opt_max(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and max attribute for all optimized independent 
        parameters (variability:parameter, free: true).
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of max 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_max', include_alias)
        vrefs = []
        max_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_variability() == PARAMETER and \
                    ftype.get_free() == True:
                        
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
            return zip(tuple(vrefs), tuple(max_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variability() == PARAMETER and \
                ftype.get_free() == True:
                    
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
        return zip(tuple(vrefs), tuple(max_values))

    def get_dx_max(self, include_alias=True, ignore_cache=False):
        """ 
        Get the max attribute of the derivatives (variable_category:derivative) 
        in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of max 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_dx_max', include_alias)
        
        vrefs = []
        max_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_variable_category() == DERIVATIVE:
                        
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
            return zip(tuple(vrefs), tuple(max_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == DERIVATIVE:
                   
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
        return zip(tuple(vrefs), tuple(max_values))

    def get_x_max(self, include_alias=True, ignore_cache=False):
        """ 
        Get the max attributes of the states (variable_category:state) in the 
        model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of max 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_max', include_alias)
        
        vrefs = []
        max_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_variable_category() == STATE:
                        
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
            return zip(tuple(vrefs), tuple(max_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == STATE:
                    
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
        return zip(tuple(vrefs), tuple(max_values))

    def get_u_max(self, include_alias=True, ignore_cache=False):
        """ 
        Get the max attributes of the inputs (variable_category:algebraic, 
        causality: input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of max 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_max', include_alias)
        
        vrefs = []
        max_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_causality() == INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                        
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
            return zip(tuple(vrefs), tuple(max_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
        return zip(tuple(vrefs), tuple(max_values))

    def get_w_max(self, include_alias=True, ignore_cache=False):
        """ 
        Get the max attributes of the algebraic variables 
        (variable_category:algebraic, causality: not input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of max 
            attribute respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_max', include_alias)
        
        vrefs = []
        max_values = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                ftype = sv.get_fundamental_type()
                if not isinstance(ftype, String) and not \
                    isinstance(ftype, Boolean) and \
                    sv.get_causality() != INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                        
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
            return zip(tuple(vrefs), tuple(max_values))
            
        for sv in scalarvariables:
            ftype = sv.get_fundamental_type()
            if not isinstance(ftype, String) and not \
                isinstance(ftype, Boolean) and \
                sv.get_alias() == NO_ALIAS and \
                sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    
                    vrefs.append(sv.get_value_reference())
                    max_values.append(ftype.get_max())
        return zip(tuple(vrefs), tuple(max_values))

    def get_p_opt_islinear(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and boolean value describing if variable appears 
        linearly in all equations and constraints for all optimized independent 
        parameters.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of is linear 
            element respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_islinear', 
                include_alias)

        vrefs = []
        is_linear = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variability() == PARAMETER and \
                    sv.get_fundamental_type().get_free() == True:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
            return zip(tuple(vrefs), tuple(is_linear))
            
        for sv in scalarvariables:
            if sv.get_variability() == PARAMETER and \
                sv.get_fundamental_type().get_free() == True:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
        return zip(tuple(vrefs), tuple(is_linear))

    def get_dx_islinear(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and boolean value describing if variable appears 
        linearly in all equations and constraints for all derivatives 
        (variable_category:derivative) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of is linear 
            element respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_dx_islinear', 
                include_alias)
        
        vrefs = []
        is_linear = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
            return zip(tuple(vrefs), tuple(is_linear))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == DERIVATIVE:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
        return zip(tuple(vrefs), tuple(is_linear))

    def get_x_islinear(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and boolean value describing if variable appears 
        linearly in all equations and constraints for all states 
        (variable_category:state) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of is linear 
            element respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_islinear', 
                include_alias)
        
        vrefs = []
        is_linear = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
            return zip(tuple(vrefs), tuple(is_linear))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == STATE:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
        return zip(tuple(vrefs), tuple(is_linear))

    def get_u_islinear(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and boolean value describing if variable appears 
        linearly in all equations and constraints for all inputs 
        (variable_category:algebraic, causality: input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of is linear 
            element respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_islinear', 
                include_alias)
        
        vrefs = []
        is_linear = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() == INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
            return zip(tuple(vrefs), tuple(is_linear))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_causality() == INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
        return zip(tuple(vrefs), tuple(is_linear))

    def get_w_islinear(self, include_alias=True, ignore_cache=False):
        """ 
        Get value reference and boolean value describing if variable appears 
        linearly in all equations and constraints for all algebraic variables 
        (variable_category:algebraic, causality: not input) in the model.
        
        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of tuples containing value reference and value of is linear 
            element respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_islinear', 
                include_alias)
        
        vrefs = []
        is_linear = []
        scalarvariables = self.get_model_variables()
        
        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() != INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
            return zip(tuple(vrefs), tuple(is_linear))
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_causality() != INPUT and \
                sv.get_variable_category() == ALGEBRAIC:
                    vrefs.append(sv.get_value_reference())
                    is_linear.append(sv.get_is_linear())
        return zip(tuple(vrefs), tuple(is_linear))
        
    def get_dx_linear_timed_variables(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get value reference and linear timed variables for all derivatives 
        (variable_category:derivative) in the model.

        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False

        Returns::
        
            A list of tuples with value reference and list of linear time 
            variables respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 
                'get_dx_linear_timed_variables', include_alias)

        tot_timepoints = []
        scalarvariables = self.get_model_variables()

        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == DERIVATIVE:
                    vref = sv.get_value_reference()
                    timepoints = []
                    
                    for tp in sv.get_is_linear_timed_variables():
                        timepoints.append(tp.get_is_linear())
                    
                    tot_timepoints.append((vref, timepoints))
            return tot_timepoints
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == DERIVATIVE:
                vref = sv.get_value_reference()
                timepoints = []
                
                for tp in sv.get_is_linear_timed_variables():
                    timepoints.append(tp.get_is_linear())
                
                tot_timepoints.append((vref, timepoints))
        return tot_timepoints
        
    def get_x_linear_timed_variables(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get value reference and linear timed variables for all states 
        (variable_category:state) in the model.

        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
    
        Returns::
        
            A list of tuples with value reference and list of linear time 
            variables respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_x_linear_timed_variables', 
                include_alias)

        tot_timepoints = []
        scalarvariables = self.get_model_variables()

        if include_alias:
            for sv in scalarvariables:
                if sv.get_variable_category() == STATE:
                    vref = sv.get_value_reference()
                    timepoints = []
                    
                    for tp in sv.get_is_linear_timed_variables():
                        timepoints.append(tp.get_is_linear())
                    
                    tot_timepoints.append((vref, timepoints))
            return tot_timepoints
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_variable_category() == STATE:
                vref = sv.get_value_reference()
                timepoints = []
                
                for tp in sv.get_is_linear_timed_variables():
                    timepoints.append(tp.get_is_linear())
                
                tot_timepoints.append((vref, timepoints))
        return tot_timepoints

    def get_u_linear_timed_variables(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get value reference and linear timed variables for all inputs 
        (variable_category:algebraic, causality: input) in the model.

        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False

        Returns::
        
            A list of tuples with value reference and list of linear time 
            variables respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_u_linear_timed_variables', 
                include_alias)

        tot_timepoints = []
        scalarvariables = self.get_model_variables()

        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() == INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                    vref = sv.get_value_reference()
                    timepoints = []
                    
                    for tp in sv.get_is_linear_timed_variables():
                        timepoints.append(tp.get_is_linear())
                    
                    tot_timepoints.append((vref, timepoints))
            return tot_timepoints
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_causality() == INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                vref = sv.get_value_reference()
                timepoints = []
                
                for tp in sv.get_is_linear_timed_variables():
                    timepoints.append(tp.get_is_linear())
                
                tot_timepoints.append((vref, timepoints))
        return tot_timepoints

    def get_w_linear_timed_variables(self, include_alias=True, 
        ignore_cache=False):
        """ 
        Get value reference and linear timed variables for all algebraic 
        variables (variable_category:algebraic, causality: not input) in the 
        model.

        Parameters::
        
            include_alias --
                If True, also include variables which are alias variables in the 
                result. If False, only non-alias variables will be included in 
                the result.
                Default: True
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False

        Returns::
        
            A list of tuples with value reference and list of linear time 
            variables respectively.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_w_linear_timed_variables', 
                include_alias)

        tot_timepoints = []
        scalarvariables = self.get_model_variables()

        if include_alias:
            for sv in scalarvariables:
                if sv.get_causality() != INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                    vref = sv.get_value_reference()
                    timepoints = []
                    
                    for tp in sv.get_is_linear_timed_variables():
                        timepoints.append(tp.get_is_linear())
                    
                    tot_timepoints.append((vref, timepoints))
            return tot_timepoints
            
        for sv in scalarvariables:
            if sv.get_alias() == NO_ALIAS and \
                sv.get_causality() != INPUT and \
                    sv.get_variable_category() == ALGEBRAIC:
                vref = sv.get_value_reference()
                timepoints = []
                
                for tp in sv.get_is_linear_timed_variables():
                    timepoints.append(tp.get_is_linear())
                
                tot_timepoints.append((vref, timepoints))
        return tot_timepoints

    def get_p_opt_value_reference(self, ignore_cache=False):
        """ 
        Get value reference for all optimized independent parameters.
        
        Parameters::
        
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            A list of value reference for all optimized independent parameters.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_p_opt_value_reference', 
                None)
                
        vrefs = []
        
        for sv in self.get_model_variables():
            if sv.get_variability() == PARAMETER and \
                sv.get_fundamental_type().get_free() == True:
                    vrefs.append(sv.get_value_reference())
        return vrefs

    def get_external_libraries(self, ignore_cache=False):
        """ 
        Get all external library entries. 
        
        Parameters::

            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The external library entries in a list.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_external_libraries', None)
        
        libraries = []
        
        tools = self.get_vendor_annotations()
        for tool in tools:
            if tool.get_name() == 'JModelica':
                annotations = tool.get_annotations()
                for annotation in annotations:
                    if annotation.get_name() == 'Library':
                        libraries.append(annotation.get_value())
        return libraries
        
    def get_external_includes(self, ignore_cache=False):
        """
        Get all external file includes.
        
        Parameters::
        
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The external file includes in a list.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_external_includes', None)
        
        includes = []
        
        tools = self.get_vendor_annotations()
        for tool in tools:
            if tool.get_name() == 'JModelica':
                annotations = tool.get_annotations()
                for annotation in annotations:
                    if annotation.get_name() == 'Include':
                        includes.append(annotation.get_value())
        return includes
        
    def get_external_lib_dirs(self, ignore_cache=False):
        """ 
        Get all external library directories. 
        
        Parameters::
        
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The external library directories in a list.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_external_lib_dirs', None)
            
        libdirs = []
        
        tools = self.get_vendor_annotations()
        for tool in tools:
            if tool.get_name() == 'JModelica':
                annotations = tool.get_annotations()
                for annotation in annotations:
                    if annotation.get_name() == 'LibraryDirectory':
                        libdirs.append(annotation.get_value())
        return libdirs
        
    def get_external_incl_dirs(self, ignore_cache=False):
        """ 
        Get all external include directories. 
        
        Parameters::
        
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The external include directories in a list.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_external_incl_dirs', None)
            
        includedirs = []
        
        tools = self.get_vendor_annotations()
        for tool in tools:
            if tool.get_name() == 'JModelica':
                annotations = tool.get_annotations()
                for annotation in annotations:
                    if annotation.get_name() == 'IncludeDirectory':
                        includedirs.append(annotation.get_value())
        return includedirs

    def get_opt_starttime(self, ignore_cache=False):
        """ 
        Get the optimization interval start time. 
        
        Parameters::
        
            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
        
        Returns::
        
            The optimization interval start time. None if there is no 
            optimization part.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_opt_starttime', None)
                
        optimization = self.get_optimization()
        
        if optimization == None:
            return None
        
        return optimization.get_interval_start_time().get_value()

    def get_opt_finaltime(self, ignore_cache=False):
        """ 
        Get the optimization interval final time. 
        
        Parameters::

            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The optimization interval final time. None if there is no 
            optimization part.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_opt_finaltime', None)
                
        optimization = self.get_optimization()
        
        if optimization == None:
            return None
        
        return optimization.get_interval_final_time().get_value()
        
    def get_opt_starttime_free(self, ignore_cache=False):
        """ 
        Get the optimization interval start time free attribute. 
        
        Parameters::

            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The optimization interval start time free attribute. None if there 
            is no optimization part.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_opt_starttime_free', None)
                
        optimization = self.get_optimization()
        
        if optimization == None:
            return None
        
        return optimization.get_interval_start_time().get_free()

    def get_opt_finaltime_free(self, ignore_cache=False):
        """ 
        Get the optimization interval final time free attribute. 
        
        Parameters::

            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The optimization interval final time free attribute. None if there 
            is no optimization part.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_opt_finaltime_free', None)
                
        optimization = self.get_optimization()
        
        if optimization == None:
            return None
        
        return optimization.get_interval_final_time().get_free()
        
    def get_opt_timepoints(self, ignore_cache=False):
        """ 
        Get the optimization time points. 
        
        Parameters::

            ignore_cache -- 
                If False look for the value in the cache first, if True skip 
                cache and derive value from data structure.
                Default: False
                
        Returns::
        
            The optimization time points. None if there is no optimization part.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_opt_timepoints', None)
                
        optimization = self.get_optimization()
        
        if optimization == None:
            return None
            
        time_points = []
        
        for tp in optimization.get_time_points():
            time_points.append(tp.get_value())
        
        return time_points
    
# ============== Here begins the XML element classes ===================
    
class BaseUnit:
    """ 
    Class defining data structure based on the XML element BaseUnit.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the BaseUnit 
        element and creates a BaseUnit object copy.
        """
        # attributes
        self._attributes = {'unit':''}
        
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
            
        # set list of display units
        self._display_unit_definitions = [];
        # fill list by calling constructor for DisplayUnitDefinition for 
        # each element
        e_unitdefs = element.getchildren()
        for e_unitdef in e_unitdefs:
            self._display_unit_definitions.append(
                DisplayUnitDefinition(e_unitdef))
            
    def get_unit(self):
        """ 
        Get the BaseUnit attribute unit.
        
        Returns::
        
            The unit attribute value as string.
        """
        return self._attributes['unit']
            
    def get_display_units(self):
        """ 
        Get all display units in the BaseUnit element.
        
        Returns::
        
            A list of all display units (type: DisplayUnitDefinition)
         """
        return self._display_unit_definitions
        
class DisplayUnitDefinition:
    """ 
    Class defining data structure based on the XML element 
    DisplayUnitDefinition.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the 
        DisplayUnitDefinition element and creates a DisplayUnitDefinition object 
        copy.
        """
        # attributes
        self._attributes = {'displayUnit':'',
                            'gain':'1.0',
                            'offset':'0.0'}
        
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_display_unit(self):
        """ 
        Get the value of the display unit attribute. 
        
        Returns::
        
            The display unit attribute value as string.
        """
        return self._attributes['displayUnit']
        
    def get_gain(self):
        """ 
        Get the value of the gain attribute.
        
        Returns::
        
            The gain attribute value as float (default: 1).
        """
        return float(self._attributes['gain'])
        
    def get_offset(self):
        """ 
        Get the value of the offset attribute.
        
        Returns::
        
            The offset attribute value as float (default: 0).
        """
        return float(self._attributes['offset'])
        
class Type:
    """ 
    Class defining data structure based on the XML element Type.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Type 
        element and creates a Type object copy.
        """
        # attributes
        self._attributes = {'name':'',
                            'description':''}
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
        
        # get fundamental type (should only be one)
        e_ftype = element.getchildren()[0]
        if e_ftype.tag == 'RealType':
            self._fundamentaltype = RealType(e_ftype)
        elif e_ftype.tag == 'IntegerType':
            self._fundamentaltype = IntegerType(e_ftype)
        elif e_ftype.tag == 'BooleanType':
            self._fundamentaltype = BooleanType(e_ftype)
        elif e_ftype.tag == 'StringType':
            self._fundamentaltype = StringType(e_ftype)
        elif e_ftype.tag == 'EnumerationType':
            self._fundamentaltype = EnumerationType(e_ftype)
        else:
            raise XMLException(
                "fundamental type (TypeDefinitions)"+str(e_ftype.tag)+\
                " is unknown")

    def get_name(self):
        """ 
        Get the value of the name attribute. 
        
        Returns::
        
            The name attribute value as string.
        """
        return self._attributes['name']
        
    def get_description(self):
        """ 
        Get the value of the description attribute. 
        
        Returns::
        
            The description attribute value as string (empty string if not set).
        """
        return self._attributes['description']
        
    def get_fundamental_type(self):
        """ 
        Get the type of the type defined. (Real, Integer, Boolean, String or 
        Enumeration Type)
        
        Returns::
        
            An object of type RealType, IntegerType, BooleanType, StringType or 
            EnumerationType.
        """
        return self._fundamentaltype
        
class RealType:
    """ 
    Class defining data structure based on the XML element RealType.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the RealType 
        element and creates a RealType object copy.
        """
        # attributes
        self._attributes = {'quantity':'',
                            'unit':'',
                            'displayUnit':'',
                            'relativeQuantity':'false',
                            'min':'',
                            'max':'',
                            'nominal':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_quantity(self):
        """ 
        Get the value of the quantity attribute.
        
        Returns::
        
            The quantity attribute value as string (empty string if not set).
        """
        return self._attributes['quantity']
    
    def get_unit(self):
        """ 
        Get the value of the unit attribute.
        
        Returns::
        
            The unit attribute value as string (empty string if not set).
        """
        return self._attributes['unit']
        
    def get_display_unit(self):
        """ 
        Get the value of the display unit attribute.
        
        Returns::
        
            The display unit attribute value as string (empty string if not 
            set).
        """
        return self._attributes['displayUnit']
        
    def get_relative_quantity(self):
        """ 
        Get the value of the relative quantity attribute.
        
        Returns::
        
            The relative quantity attribute value as bool (default: false).
        """
        return _translate_xmlbool(self._attributes['relativeQuantity'])
    
    def get_min(self):
        """ 
        Get the value of the min attribute.
        
        Returns::
        
            The min attribute value as float (None if not set).
        """
        if self._attributes['min'] == '':
            return None
        return float(self._attributes['min'])
        
    def get_max(self):
        """ 
        Get the value of the max attribute.
        
        Returns::
        
            The max attribute value as float (None if not set).
        """
        if self._attributes['max'] == '':
            return None
        return float(self._attributes['max'])
        
    def get_nominal(self):
        """ 
        Get the value of the nominal attribute.
        
        Returns::
        
            The nominal attribute value as float (None if not set).
        """
        if self._attributes['nominal'] == '':
            return None
        return float(self._attributes['nominal'])
        
class IntegerType:
    """ 
    Class defining data structure based on the XML element IntegerType.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the IntegerType 
        element and creates a IntegerType object copy.
        """
        self._attributes = {'quantity':'',
                            'min':'',
                            'max':''}
    
    def get_quantity(self):
        """ 
        Get the value of the quantity attribute.
        
        Returns::
        
            The quantity attribute value as string (empty string if not set).
        """
        return self._attributes['quantity']
        
    def get_min(self):
        """ 
        Get the value of the min attribute.
        
        Returns::
        
            The min attribute value as int (None if not set).
        """
        if self._attributes['min'] == '':
            return None
        return int(self._attributes['min'])
        
    def get_max(self):
        """ 
        Get the value of the max attribute.
        
        Returns::
        
            The max attribute value as int (None if not set).
        """
        if self._attributes['max'] == '':
            return None
        return int(self._attributes['max'])

class BooleanType:
    """ 
    Class defining data structure based on the XML element BooleanType. Is empty 
    since XML element contains no attributes or other elements.
    """
    def __init__(self, element):
        pass


class StringType:
    """ 
    Class defining data structure based on the XML element StringType. Is empty 
    since XML element contains no attributes or other elements.
    """
    def __init__(self, element):
        pass

    
class EnumerationType:
    """ 
    Class defining data structure based on the XML element EnumerationType.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the 
        EnumerationType element and creates a EnumerationType object copy.
        """
        self._attributes = {'quantity':'',
                            'min':'',
                            'max':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

        # items
        self._items = []
        e_items = element.getchildren()
        for e_item in e_items:
            self._items.append(Item(e_item))
                            
    def get_quantity(self):
        """ 
        Get the value of the quantity attribute.
        
        Returns::
        
            The quantity attribute value as string (empty string if not set).
        """
        return self._attributes['quantity']
        
    def get_min(self):
        """ 
        Get the value of the min attribute.
        
        Returns::
        
            The min attribute value as int (None if not set).
        """
        if self._attributes['min'] == '':
            return None
        return int(self._attributes['min'])
        
    def get_max(self):
        """ 
        Get the value of the max attribute.
        
        Returns::
        
            The max attribute value as int (None if not set).
        """
        if self._attributes['max'] == '':
            return None
        return int(self._attributes['max'])
        
    def get_items(self):
        """ 
        Get the items defined in the enumeration.
        
        Returns::
        
            A list of all items (type: Item)
        """
        return self._items

class Item:
    """ 
    Class defining data structure based on the XML element Item.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Item 
        element and creates a Item object copy.
        """
        self._attributes = {'name':'',
                            'description':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
        
    def get_name(self):
        """ 
        Get the value of the name attribute.
        
        Returns::
        
            The name attribute value as string.
        """
        return self._attributes['name']
        
    def get_description(self):
        """ 
        Get the value of the description attribute.
        
        Returns::
        
            The description attribute value as string.
        """
        return self._attributes['description']

class DefaultExperiment:
    """ 
    Class defining data structure based on the XML element DefaultExperiment.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the 
        DefaultExperiment element and creates a DefaultExperiment object copy.
        """
        self._attributes = {'startTime':'',
                            'stopTime':'',
                            'tolerance':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
    
    def get_start_time(self):
        """ 
        Get the value of the start time attribute.
        
        Returns::
        
            The start time attribute value as float (None if not set).
        """
        if self._attributes['startTime'] == '':
            return None
        return float(self._attributes['startTime'])
        
    def get_stop_time(self):
        """ 
        Get the value of the stop time attribute.
        
        Returns::
        
            The stop time attribute value as float (None if not set).
        """
        if self._attributes['stopTime'] == '':
            return None
        return float(self._attributes['stopTime'])
        
    def get_tolerance(self):
        """ 
        Get the value of the tolerance attribute.
        
        Returns::
        
            The tolerance attribute value as float (None if not set).
        """
        if self._attributes['tolerance'] == '':
            return None
        return float(self._attributes['tolerance'])
        
class Tool:
    """ 
    Class defining data structure based on the XML element Tool.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Tool 
        element and creates a Tool object copy.
        """
        self._attributes = {'name':''}
        
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
        
        # annotations
        self._annotations = []
        e_annotations = element.getchildren()
        for e_annotation in e_annotations:
            self._annotations.append(Annotation(e_annotation))
        
    def get_name(self):
        """ 
        Get the value of the name attribute.
        
        Returns::
        
            The name attribute value as string.
        """
        return self._attributes['name']

    def get_annotations(self):
        """ 
        Get all annotations set for the tool.
        
        Returns::
        
            A list of all annotations (type: Annotation)
        """
        return self._annotations
        
class Annotation:
    """ 
    Class defining data structure based on the XML element Annotation.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Annotation 
        element and creates a Annotation object copy.
        """
        self._attributes = {'name':'',
                            'value':''}
        
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_name(self):
        """ 
        Get the value of the name attribute.
        
        Returns::
        
            The name attribute value as string.
        """
        return self._attributes['name']
        
    def get_value(self):
        """ 
        Get the value of the value attribute.
        
        Returns::
        
            The value attribute value as string.
        """
        return self._attributes['value']
        
class ScalarVariable:
    """ 
    Class defining data structure based on the XML element ScalarVariable.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the 
        ScalarVariable element and creates a ScalarVariable object copy.
        """
        self._attributes = {'name':'',
                            'valueReference':'',
                            'description':'',
                            'variability':'continuous',
                            'causality':'internal',
                            'alias':'noAlias'}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
 
        # get fundamental type (must be one of Real, Integer, Boolean, String, Enumeration)
        e_ftype = element.getchildren()[0]
        if e_ftype.tag == 'Real':
            self._fundamental_type = Real(e_ftype)
        elif e_ftype.tag == 'Integer':
            self._fundamental_type = Integer(e_ftype)
        elif e_ftype.tag == 'Boolean':
            self._fundamental_type = Boolean(e_ftype)
        elif e_ftype.tag == 'String':
            self._fundamental_type = String(e_ftype)
        elif e_ftype.tag == 'Enumeration':
            self._fundamental_type = Enumeration(e_ftype)
        else:
            raise XMLException("ScalarVariable: "+self._attributes['name']+
                " does not have a valid fundamental type.")

        # direct dependency
        self._direct_dependency = None
        e_directdependency = element.find('DirectDependency')
        if e_directdependency != None:
            self._direct_dependency = DirectDependency(e_directdependency)
            
        #### Qualified Name here ####
        
        # isLinear
        self._is_linear = element.find('isLinear')
        if self._is_linear != None:
            self._is_linear = _translate_xmlbool(self._is_linear.text)
            
        # isLinearTimedVariables
        self._is_linear_timed_variables = []
        e_lineartimedvariables = element.find('isLinearTimedVariables')
        if e_lineartimedvariables != None:
            e_tpoints = e_lineartimedvariables.getchildren()
            for e_tp in e_tpoints:
                self._is_linear_timed_variables.append(TimePoint(e_tp))
                
        # variableCategory
        e_variablecategory = element.find('VariableCategory')
        if e_variablecategory == None:
            self._variable_category = 'algebraic'
        else:
            self._variable_category = e_variablecategory.text

    def get_name(self):
        """ 
        Get the value of the name attribute.
        
        Returns::
        
            The name attribute value as string.
        """
        return self._attributes['name']
        
    def get_value_reference(self):
        """ 
        Get the value of the value reference attribute.
        
        Returns::
        
            The value reference as unsigned int.
        """
        if self._attributes['valueReference'] == '':
            return None
        return uint(self._attributes['valueReference'])
        
    def get_description(self):
        """ 
        Get the value of the description attribute.
        
        Returns::
        
            The description attribute value as string (empty string if not set).
        """
        return self._attributes['description']
        
    def get_variability(self):
        """ 
        Get the value of the variability attribute.
        
        Returns::
        
            The variability attribute value as enumeration: CONTINUOUS, 
            CONSTANT, PARAMETER or DISCRETE (default: CONTINUOUS).
        """
        return _translate_variability(self._attributes['variability'])
        
    def get_causality(self):
        """ 
        Get the value of the causality attribute.
        
        Returns::
        
            The causality attribute value as enumeration: INTERNAL, INPUT, 
            OUTPUT or NONE. (default: INTERNAL).
        """
        return _translate_causality(self._attributes['causality'])
        
    def get_alias(self):
        """ 
        Get the value of the alias attribute.
        
        Returns::
        
            The alias attribute value as enumeration: NO_ALIAS, ALIAS or 
            NEGATED_ALIAS. (default: NO_ALIAS).
        """
        return _translate_alias(self._attributes['alias'])
        
    def get_fundamental_type(self):
        """ 
        Get the type of the type defined. (Real, Integer, Boolean, String or 
        Enumeration Type)
        
        Returns::
        
            An object of type Real, Integer, Boolean, String or Enumeration.
        """
        return self._fundamental_type
        
    def get_direct_dependency(self):
        """ 
        Get the direct dependencies set for the variable.
        
        Returns::
        
            An object of type DirectDependency (None if not set).
        """
        return self._direct_dependency
        
    def get_is_linear(self):
        """ 
        Get the value of the is linear element.
        
        Returns::
        
            The is linear attribute value as bool (None if not set).
        """
        return self._is_linear
        
    def get_is_linear_timed_variables(self):
        """ 
        Get all is linear timed variables set for the variable.
        
        Returns::
        
            A list of all linear timed variables (type: TimePoint)
        """
        return self._is_linear_timed_variables
    
    def get_variable_category(self):
        """ 
        Get the value of the variable category element.
        
        Returns::
        
            The variable category attribute value as enumeration: 
            ALGEBRAIC, STATE, DEPENDENT_CONSTANT, INDEPENDENT_CONSTANT, 
            DEPENDENT_PARAMETER, INDEPENDENT_PARAMETER or DERIVATIVE  
            (default: ALGEBRAIC).
        """
        return _translate_variable_category(self._variable_category)
        
class Real:
    """ 
    Class defining data structure based on the XML element Real.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Real 
        element and creates a Real object copy.
        """
        self._attributes = {'declaredType':'',
                            'quantity':'',
                            'unit':'',
                            'displayUnit':'',
                            'relativeQuantity':'false',
                            'min':'',
                            'max':'',
                            'nominal':'',
                            'start':'',
                            'fixed':'',
                            'free':'',
                            'initialGuess':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_declared_type(self):
        """ 
        Get the value of the declared type attribute.
        
        Returns::
        
            The declared type attribute value as string (empty string if not 
            set).
        """
        return self._attributes['declaredType']
        
    def get_quantity(self):
        """ 
        Get the value of the quantity attribute.
        
        Returns::
        
            The quantity attribute value as string (empty string if not set).
        """
        return self._attributes['quantity']
        
    def get_unit(self):
        """ 
        Get the value of the unit attribute.
        
        Returns::
        
            The unit attribute value as string (empty string if not set).
        """
        return self._attributes['unit']
        
    def get_display_unit(self):
        """ 
        Get the value of the display unit attribute.
        
        Returns::
        
            The display unit attribute value as string (empty string if not 
            set).
        """
        return self._attributes['displayUnit']
        
    def get_relative_quantity(self):
        """ 
        Get the value of the relative quantity attribute.
        
        Returns::
        
            The relative quantity attribute value as bool (default: false).
        """
        return _translate_xmlbool(self._attributes['relativeQuantity'])
        
    def get_min(self):
        """ 
        Get the value of the min attribute.
        
        Returns::
        
            The min attribute value as float (None if not set).
        """
        min = self._attributes['min']
        if min == '':
            return None
        return float(min)
        
    def get_max(self):
        """ 
        Get the value of the max attribute.
        
        Returns::
        
            The max attribute value as float (None if not set).
        """
        max = self._attributes['max']
        if max == '':
            return None
        return float(max)

    def get_nominal(self):
        """ 
        Get the value of the nominal attribute.
        
        Returns::
        
            The nominal attribute value as float (None if not set).
        """
        nominal = self._attributes['nominal']
        if nominal == '':
            return None
        return float(nominal)

    def get_start(self):
        """ 
        Get the value of the start attribute.
        
        Returns::
        
            The start attribute value as float (None if not set).
        """
        start = self._attributes['start']
        if start == '':
            return None
        return float(start)

    def get_fixed(self):
        """ 
        Get the value of the fixed attribute.
        
        Returns::
        
            The fixed attribute value as bool (None if not set).
        """
        fixed = self._attributes['fixed']
        if fixed == '':
            return None
        return _translate_xmlbool(fixed)
        
    def get_free(self):
        """ 
        Get the value of the free attribute.
        
        Returns::
        
            The free attribute value as bool (None if not set).
        """
        free = self._attributes['free']
        if free == '':
            return None
        return _translate_xmlbool(free)

    def get_initial_guess(self):
        """ 
        Get the value of the attribute.
        
        Returns::
        
            The initial guess attribute value as float (None if not set).
        """
        initialguess = self._attributes['initialGuess']
        if initialguess == '':
            return None
        return float(initialguess)
        
class Integer:
    """ 
    Class defining data structure based on the XML element Integer.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Integer 
        element and creates a Integer object copy.
        """
        self._attributes = {'declaredType':'',
                            'quantity':'',
                            'min':'',
                            'max':'',
                            'start':'',
                            'fixed':'',
                            'free':'',
                            'initialGuess':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_declared_type(self):
        """ 
        Get the value of the declared type attribute.
        
        Returns::
        
            The declared type attribute value as string (empty string if not 
            set).
        """
        return self._attributes['declaredType']
        
    def get_quantity(self):
        """ 
        Get the value of the quantity attribute.
        
        Returns::
        
            The quantity attribute value as string (empty string if not set).
        """
        return self._attributes['quantity']
                
    def get_min(self):
        """ 
        Get the value of the min attribute.
        
        Returns::
        
            The min attribute value as int (None if not set).
        """
        min = self._attributes['min']
        if min == '':
            return None
        return int(min)
        
    def get_max(self):
        """ 
        Get the value of the max attribute.
        
        Returns::
        
            The max attribute value as int (None if not set).
        """
        max = self._attributes['max']
        if max == '':
            return None
        return int(max)

    def get_start(self):
        """ 
        Get the value of the start attribute.
        
        Returns::
        
            The start attribute value as int (None if not set).
        """
        start = self._attributes['start']
        if start == '':
            return None
        return int(start)

    def get_fixed(self):
        """ 
        Get the value of the fixed attribute.
        
        Returns::
        
            The fixed attribute value as bool (None if not set).
        """
        fixed = self._attributes['fixed']
        if fixed == '':
            return None
        return _translate_xmlbool(fixed)
        
    def get_free(self):
        """ 
        Get the value of the free attribute.
        
        Returns::
        
            The free attribute value as bool (None if not set).
        """
        free = self._attributes['free']
        if free == '':
            return None
        return _translate_xmlbool(free)

    def get_initial_guess(self):
        """ 
        Get the value of the attribute.
        
        Returns::
        
            The initial guess attribute value as int (None if not set).
        """
        initialguess = self._attributes['initialGuess']
        if initialguess == '':
            return None
        return int(initialguess)
        
class Boolean:
    """ 
    Class defining data structure based on the XML element Boolean.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Boolean 
        element and creates a Boolean object copy.
        """
        self._attributes = {'declaredType':'',
                            'start':'',
                            'fixed':'',
                            'free':'',
                            'initialGuess':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_declared_type(self):
        """ 
        Get the value of the declared type attribute.
        
        Returns::
        
            The declared type attribute value as string (empty string if not 
            set).
        """
        return self._attributes['declaredType']
        
    def get_start(self):
        """ 
        Get the value of the start attribute.
        
        Returns::
        
            The start attribute value as bool (None if not set).
        """
        start = self._attributes['start']
        if start == '':
            return None
        return _translate_xmlbool(start)

    def get_fixed(self):
        """ 
        Get the value of the fixed attribute.
        
        Returns::
        
            The fixed attribute value as bool (None if not set).
        """
        fixed = self._attributes['fixed']
        if fixed == '':
            return None
        return _translate_xmlbool(fixed)
        
    def get_free(self):
        """ 
        Get the value of the free attribute.
        
        Returns::
        
            The free attribute value as bool (None if not set).
        """
        free = self._attributes['free']
        if free == '':
            return None
        return _translate_xmlbool(free)

    def get_initial_guess(self):
        """ 
        Get the value of the attribute.
        
        Returns::
        
            The initial guess attribute value as bool (None if not set).
        """
        initialguess = self._attributes['initialGuess']
        if initialguess == '':
            return None
        return _translate_xmlbool(initialguess)

class String:
    """ 
    Class defining data structure based on the XML element String.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the String 
        element and creates a String object copy.
        """
        self._attributes = {'declaredType':'',
                            'start':'',
                            'fixed':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_declared_type(self):
        """ 
        Get the value of the declared type attribute.
        
        Returns::
        
            The declared type attribute value as string (empty string if not 
            set).
        """
        return self._attributes['declaredType']
        
    def get_start(self):
        """ 
        Get the value of the start attribute.
        
        Returns::
        
            The start attribute value as string (None if not set).
        """
        return self._attributes['start']

    def get_fixed(self):
        """ 
        Get the value of the fixed attribute.
        
        Returns::
        
            The fixed attribute value as bool (None if not set).
        """
        fixed = self._attributes['fixed']
        if fixed == '':
            return None
        return _translate_xmlbool(fixed)


class Enumeration:
    """ 
    Class defining data structure based on the XML element Enumeration.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the Enumeration 
        element and creates a Enumeration object copy.
        """
        self._attributes = {'declaredType':'',
                            'quantity':'',
                            'min':'',
                            'max':'',
                            'start':'',
                            'free':'',
                            'fixed':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_declared_type(self):
        """ 
        Get the value of the declared type attribute.
        
        Returns::
        
            The declared type attribute value as string.
        """
        return self._attributes['declaredType']
        
    def get_quantity(self):
        """ 
        Get the value of the quantity attribute.
        
        Returns::
        
            The quantity attribute value as string (empty string if not 
            set).
        """
        return self._attributes['quantity']

    def get_min(self):
        """ 
        Get the value of the min attribute.
        
        Returns::
        
            The min attribute value as int (None if not set).
        """
        min = self._attributes['min']
        if min == '':
            return None
        return int(min)
        
    def get_max(self):
        """ 
        Get the value of the max attribute.
        
        Returns::
        
            The max attribute value as int (None if not set).
        """
        max = self._attributes['max']
        if max == '':
            return None
        return int(max)

    def get_start(self):
        """ 
        Get the value of the start attribute.
        
        Returns::
        
            The start attribute value as int (None if not set).
        """
        start = self._attributes['start']
        if start == '':
            return None
        return int(start)
        
    def get_free(self):
        """ 
        Get the value of the free attribute.
        
        Returns::
        
            The free attribute value as bool (None if not set).
        """
        free = self._attributes['free']
        if free == '':
            return None
        return _translate_xmlbool(free)

    def get_fixed(self):
        """ 
        Get the value of the fixed attribute.
        
        Returns::
        
            The fixed attribute value as bool (None if not set).
        """
        fixed = self._attributes['fixed']
        if fixed == '':
            return None
        return _translate_xmlbool(fixed)

class DirectDependency:
    """ 
    Class defining data structure based on the XML element DirectDependency.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the 
        DirectDependency element and creates a DirectDependency object copy.
        """
        self._names = []
        e_names = element.getchildren()
        for e_name in e_names:
            self._names.append(e_name.text)
            
    def get_names(self):
        """ 
        Get the names of the input variables needed to compute this output.
        
        Returns::
        
                A list of variable names as string.
        """
        return self._names

class TimePoint:
    """ 
    Class defining data structure based on the XML element TimePoint.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the TimePoint 
        element and creates a TimePoint object copy.
        """
        self._attributes = {'index':'',
                            'isLinear':''}
                            
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)

    def get_index(self):
        """ 
        Get the value of the index attribute.
        
        Returns::
        
            The index attribute value as int.
        """
        return int(self._attributes['index'])
            
    def get_is_linear(self):
        """ 
        Get the value of the is linear attribute.
        
        Returns::
        
            The is linear attribute value as bool.
        """
        return _translate_xmlbool(self._attributes['isLinear'])
            
class Optimization:
    """ 
    Class defining data structure based on the XML element Optimization.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the 
        Optimization element and creates an Optimization object copy.
        """
        self._attributes = {'static':''}

        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
        
        # namespace
        opt=element.nsmap['opt']
        ns="{"+opt+"}"        
        
        # interval start time
        self._interval_start_time = None
        e_intervalstartt = element.find(ns+'IntervalStartTime')
        if e_intervalstartt != None:
            self._interval_start_time = Opt_IntervalTime(e_intervalstartt)
            
        # interval final time
        self._interval_final_time = None
        e_intervalfinalt = element.find(ns+'IntervalFinalTime')
        if e_intervalfinalt != None:
            self._interval_final_time = Opt_IntervalTime(e_intervalfinalt)
            
        # time points
        # bad xml schema construction - consider redoing
        self._time_points = []
        e_timepoints = element.find(ns+'TimePoints')
        if e_timepoints != None:
            e_indexes = e_timepoints.findall(ns+"Index")
            e_values = e_timepoints.findall(ns+"Value")
            
            for i, e_index in enumerate(e_indexes):
                self._time_points.append(Opt_TimePoint(e_index, e_values[i]))
                
    def get_static(self):
        """ 
        Get the value of the static attribute.
        
        Returns::
        
            The static attribute value as bool (None if not set).
        """
        if self._attributes['static'] == '':
            return None
        return _translate_xmlbool(self._attributes['static'])
            
    def get_interval_start_time(self):
        """ 
        Get the interval start time set for this model.
        
        Returns::
        
            An object of type Opt_IntervalTime containing start time data (None 
            if not set).
        """
        return self._interval_start_time
        
    def get_interval_final_time(self):
        """ 
        Get the interval final time set for this model.
        
        Returns::
        
            An object of type Opt_IntervalTime containing final time data (None 
            if not set).
        """
        return self._interval_final_time
        
    def get_time_points(self):
        """ 
        Get the optimization time points set for this model.
        
        Returns::
        
            A list of all time points (type: Opt_TimePoint).
        """
        return self._time_points
        
class Opt_IntervalTime:
    """ 
    Class defining data structure based on the XML element OptIntervalFinalTime 
    and OptIntervalStartTime.
    """
    def __init__(self, element):
        """ 
        Constructor which takes an XML element object describing the ns:opt 
        IntervalStartTime and IntervalFinalTime elements and creates an 
        Opt_IntervalTime object with start or final time data.
        """
        opt=element.nsmap['opt']
        ns="{"+opt+"}"
        
        # value
        e_value = element.find(ns+"Value")
        if e_value != None:
            self._value = float(e_value.text)
        else:
            self._value = None

        # free
        e_free = element.find(ns+"Free")
        if e_free != None:
            self._free = _translate_xmlbool(e_free.text)
        else:
            self._free = None
            
        # initial guess
        e_initialguess = element.find(ns+"InitialGuess")
        if e_initialguess != None:
            self._initial_guess = float(e_initialguess.text)
        else:
            self._initial_guess = None
            
    
    def get_value(self):
        """ 
        Get the value of the value element.
        
        Returns::
        
            The value attribute value as float (None if not set).
        """
        return self._value

    def get_free(self):
        """ 
        Get the free of the value element.
        
        Returns::
        
            The free attribute value as bool (None if not set).
        """
        return self._free
        
    def get_initial_guess(self):
        """ 
        Get the value of the initial guess element.
        
        Returns::
        
            The initial guess attribute value as float (None if not set).
        """
        return self._initial_guess
    
class Opt_TimePoint:
    """ 
    Class defining data structure based on the XML element TimePoint.
    """
    def __init__(self, e_index, e_value):
        """ 
        Constructor which takes an XML element object describing the ns:opt 
        TimePoint element and creates an Opt_TimePoint object with time point 
        data.
        """
        self._index = e_index.text
        self._value = e_value.text
        
    def get_index(self):
        """ 
        Get the value of the index element.
        
        Returns::
        
            The index attribute value as int.
        """
        return int(self._index)
        
    def get_value(self):
        """ 
        Get the value of the value element.
        
        Returns::
        
            The value attribute value as float.
        """
        return float(self._value)
        

#=======================================================================

class IndependentParameters:
    
    def __init__(self, filename, schemaname=''):
        """
        Create an XML document object representation.
        
        Parse an XML document and create a full XML document object 
        representation. Validate against XML schema before parsing if the 
        parameter schemaname is set.
        
        Parameters::
        
            filename --
                The name of the XML file to parse.
                
            schemaname --
                The name of the XSD file to validate against.
                Default: Empty string (no validation).
        """
        
        # set up cache, parse XML file and obtain the root
        self.function_cache = XMLFunctionCache()
        self._element_tree = _parse_XML(filename, schemaname)
        root = self._element_tree.getroot()
        
        # build internal data structure from XML file
        self._parse_element_tree(root)
        
    def _parse_element_tree(self, root):
        """ 
        Helper function. Parse the XML element tree and build up internal data 
        structure. 
        
        Parameters::
        
            root -- 
                Reference to the root of the element tree.
        """
        self._fill_parameters(root)

    def _fill_parameters(self, root):
        """
        Helper function. Fill the internal structure of parameters with values 
        from the parsed XML file.
        """
        
        self._indep_parameters = []
        
        reals = root.findall('RealParameter')
        for r in reals:
            self._indep_parameters.append(RealParameter(r))
        
        ints = root.findall('IntegerParameter')
        for i in ints:
            self._indep_parameters.append(IntegerParameter(i))
            
        bools = root.findall('BooleanParameter')
        for b in bools:
            self._indep_parameters.append(BooleanParameter(b))
            
        strings = root.findall('StringParameter')
        for s in strings:
            self._indep_parameters.append(StringParameter(s))
            
        enums = root.findall('EnumParameter')
        for e in enums:
            self._indep_parameters.append(EnumParameter(e))
            
    def get_iparam_values(self, ignore_cache=False):
        """ 
        Extract name and value for all independent parameters in the XML 
        document.
        
        Returns::
           
            A dict with variable name as key and parameter as value.
        """
        if not ignore_cache:
            return self.function_cache.get(self, 'get_iparam_values', None)
            
        names = []
        values = []
        for ip in self._indep_parameters:
            names.append(ip.get_name())
            values.append(ip.get_value())
            
        return dict(zip(names, values))
        
    def get_all_parameters(self):
        """
        Get a list of all parameters in this XML representation.
        
        Returns::
        
            A list of all parameters.
        """
        return self._indep_parameters
        
    def write_to_file(self, filename):
        """
        Create a new XML file of the document representation.

        Parameters::
            
            filename --
                Full path of the file to create.
        """
        self._element_tree.write(filename)
            

class Parameter:
    """
    Class representing an independent parameter in a IndependentParameters XML 
    document representation.
    """
    
    def __init__(self, element):
        """
        Create a new parameter representation. (Abstract class)
        """
        self._attributes = {'name':'','value':None}
        # update attribute dict with attributes from XML file
        self._attributes.update(element.attrib)
        self._element = element
        
    def get_name(self):
        """
        Get the name of the parameter.
        
        Returns::
        
            The name of the parameter.
        """
        return self._attributes['name']
        
    def get_value(self):
        """
        Get the value of the parameter.
        
        Abstract method - must be implemented by extending classes.
        """
        pass
        
    def set_value(self, value):
        """
        Set the value of the parameter.
        
        Abstract method - must be implemented by extending classes.
        """
        pass
        
class RealParameter(Parameter):
    """
    Class representing an independent real parameter in a IndependentParameters 
    XML document representation.
    """
    
    def get_value(self):
        """
        Get the value of the parameter.
        
        Returns::
        
            The value of the parameter as real.
        """
        return float(self._attributes['value'])
        
    def set_value(self, value):
        """
        Set the value of the parameter.
        
        Parameters::
            
            value --
                The new value of the parameter.
        """
        self._attributes.update({'value':float(value)})
        self._element.set('value',str(value))
    
class IntegerParameter(Parameter):
    """
    Class representing an independent integer parameter in a 
    IndependentParameters XML document representation.
    """
    
    def get_value(self):
        """
        Get the value of the parameter.
        
        Returns::
        
            The value of the parameter as integer.
        """
        return int(self._attributes['value'])
        
    def set_value(self, value):
        """
        Set the value of the parameter.
        
        Parameters::
            
            value --
                The new value of the parameter.
        """
        self._attributes.update({'value':int(value)})
        self._element.set('value',str(value))
    
class BooleanParameter(Parameter):
    """
    Class representing an independent boolean parameter in a 
    IndependentParameters XML document representation.
    """
    
    def get_value(self):
        """
        Get the value of the parameter.
        
        Returns::
        
            The value of the parameter as boolean.
        """
        return _translate_xmlbool(self._attributes['value'])
        
    def set_value(self, value):
        """
        Set the value of the parameter.
        
        Parameters::
            
            value --
                The new value of the parameter.
        """
        self._attributes.update({'value':_translate_xmlbool(value)})
        self._element.set('value',str(value))
    
class StringParameter(Parameter):
    """
    Class representing an independent string parameter in a 
    IndependentParameters XML document representation.
    """
    
    def get_value(self):
        """
        Get the value of the parameter.
        
        Returns::
        
            The value of the parameter as string.
        """
        return str(self._attributes['value'])

    def set_value(self, value):
        """
        Set the value of the parameter.
        
        Parameters::
            
            value --
                The new value of the parameter.
        """
        self._attributes.update({'value':str(value)})
        self._element.set('value',str(value))
    
class EnumParameter(Parameter):
    """
    Class representing an independent enumeration parameter in a 
    IndependentParameters XML document representation.
    """
    
    def get_value(self):
        """
        Get the value of the parameter.
        
        Returns::
        
            The value of the parameter as integer.
        """
        return int(self._attributes['value'])
        
    def set_value(self, value):
        """
        Set the value of the parameter.
        
        Parameters::
            
            value --
                The new value of the parameter.
        """
        self._attributes.update({'value':int(value)})
        self._element.set('value',str(value))
        
#===================== FMI 2.0 ========================================

class ModelDescription2 (ModelDescription):
    """
    A class for storing the ModelDescription information in FMI 2.0.
    """
    
    def _parse_element_tree(self, root):
        """
        Override the default _parse_element_tree function to handle
        FMI 2.0 specific elements.
        """
        ModelDescription._parse_element_tree(self, root)
    
        # Build the ModelStructure data structures
        self._fill_model_structure(root)
        
    def _fill_model_structure(self, root):
        """ 
        Create the model structure data structure and fill with 
        data from the XML file.
        """
        self._model_structure_inputs = []
        self._model_structure_inputs_dict = {}

        self._model_structure_derivatives = []
        self._model_structure_derivatives_dict = {}

        self._model_structure_outputs = []
        self._model_structure_outputs_dict = {}
        
        try:
            e_modelstructure = root.find('ModelStructure')
        except:
            return
        
        inputs = e_modelstructure[0]
        derivatives = e_modelstructure[1]
        outputs = e_modelstructure[2]
        
        for e_input in inputs.getchildren():
            inp = MSInput(e_input)
            self._model_structure_inputs.append(inp)
            self._model_structure_inputs_dict[inp.get_name()] = inp

        for e_derivative in derivatives.getchildren():
            deriv = MSDerivative(e_derivative)
            self._model_structure_derivatives.append(deriv)
            self._model_structure_derivatives_dict[deriv.get_name()] = deriv

        for e_output in outputs.getchildren():
            outp = MSOutput(e_output)
            self._model_structure_outputs.append(outp)
            self._model_structure_outputs_dict[outp.get_name()] = outp
            
            
    def get_number_of_inputs(self):
        """
        Get the number of inputs.
        
        Returns::
        
            The number of inputs.
        """
        return len(self._model_structure_inputs)
    
    def get_number_of_continuous_inputs(self):
        """
        Get the number of continuous inputs.
        
        Returns::
        
            The number of continuous inputs.
        """
        return len(self.get_continous_inputs_value_references())
    
    def get_number_of_outputs(self):
        """
        Get the number of outputs.
        
        Returns::
        
            The number of outputs.
        """
        return len(self._model_structure_inputs)
    
    def get_number_of_continuous_outputs(self):
        """
        Get the number of continuous outputs.
        
        Returns::
        
            The number of continuous outputs.
        """
        return len(self.get_continous_outputs_value_references())
            
    def get_continous_outputs_value_references(self):
        """
        Get the value references of the of continuous outputs.
        
        Returns::
        
            A list of value references.
        """    
        yc_vrefs=[]        
        for vv in self._model_structure_outputs:
            v = self._model_variables_dict[vv.get_name()]
            if v.get_variability()==CONTINUOUS:
                yc_vrefs.append(v.get_value_reference())
        return yc_vrefs

    def get_continous_inputs_value_references(self):
        """
        Get the value references of the of continuous inputs.
        
        Returns::
        
            A list of value references.
        """
        uc_vrefs=[]        
        for vv in self._model_structure_inputs:
            v = self._model_variables_dict[vv.get_name()]
            if v.get_variability()==CONTINUOUS:
                uc_vrefs.append(v.get_value_reference())
        return uc_vrefs

    def get_state_dependency(self, variable):
        """
        Get the state dependencies of a particular derivative or
        output variable.
        
        Returns::
        
            A list of state indices (0-indexing).
        """
        try:
            v = self._model_structure_derivatives_dict[variable]
            return v.get_state_dependency()
        except:
            try:
                v = self._model_structure_outputs_dict[variable]
                return v.get_state_dependency()
            except:
                raise Exception("No state dependency information is available for variable " + variable)

    def get_input_dependency(self, variable):
        """
        Get the input dependencies of a particular derivative or
        output variable.
        
        Returns::
        
            A list of input indices (0-indexing).
        """

        try:
            v = self._model_structure_derivatives_dict[variable]
            return v.get_input_dependency()
        except:
            try:
                v = self._model_structure_outputs_dict[variable]
                return v.get_input_dependency()
            except:
                raise Exception("No input dependency information is available for variable " + variable)


class MSVariable:
    """
    Base class describing a variable in the ModelStructure section.
    """
    def __init__(self,element):
        """
        Constructor that takes an XML element as input.
        """
        self._name = element.attrib['name']

    def get_name(self):
        """
        Get the name of the variable.
        
        Returns::
        
            The name of the variable.
        """
        return self._name

class MSInput(MSVariable):
    """
    Class describing an input variable in the ModelStructure section.
    """

    pass
    
class MSDerivative(MSVariable):
    """
    Class describing a derivative variable in the ModelStructure section.
    """
    
    def __init__(self,element):
        """
        Constructor taking an XML element as argument.
        """
        MSVariable.__init__(self, element)
        try:
            self._state_dependency = element.attrib['stateDependency'].split()  
            for i in range(len(self._state_dependency)):
                self._state_dependency[i] = int(self._state_dependency[i]) - 1
        except:
            self._state_dependency = []
        try:
            self._input_dependency = element.attrib['inputDependency'].split()      
            for i in range(len(self._input_dependency)):
                self._input_dependency[i] = int(self._input_dependency[i]) - 1            
        except:
            self._input_dependency = []      
        
    def get_state(self):
        """
        Get the name of the state corresponding to the derivative.
        
        Returns::
        
            The name of the state.
        """
        return self._state

    def get_state_dependency(self):
        """
        Get the state dependency. The variable depends directly on
        the states corresponding to the returned list of indices.
        
        Returns::
        
            A list of state indices.
        """
        return self._state_dependency

    def get_input_dependency(self):
        """
        Get the input dependency. The variable depends directly on
        the states corresponding to the returned list of indices.
        
        Returns::
        
            A list of input indices.
        """
        return self._input_dependency

class MSOutput(MSVariable):
    """
    Class describing a derivative variable in the ModelStructure section.
    """
        
    def __init__(self,element):
        MSVariable.__init__(self, element)
        try:
            self._state_dependency = element.attrib['stateDependency'].split()  
            for i in range(len(self._state_dependency)):
                self._state_dependency[i] = int(self._state_dependency[i]) - 1
        except:
            self._state_dependency = []
        try:
            self._input_dependency = element.attrib['inputDependency'].split()      
            for i in range(len(self._input_dependency)):
                self._input_dependency[i] = int(self._input_dependency[i]) - 1            
        except:
            self._input_dependency = []      

    def get_state_dependency(self):
        """
        Get the state dependency. The variable depends directly on
        the states corresponding to the returned list of indices.
        
        Returns::
        
            A list of state indices.
        """

        return self._state_dependency

    def get_input_dependency(self):
        """
        Get the input dependency. The variable depends directly on
        the states corresponding to the returned list of indices.
        
        Returns::
        
            A list of input indices.
        """
        return self._input_dependency
        
#=======================================================================

class XMLFunctionCache:
    """ 
    Class representing cache for loaded XML doc.
    
    Function return values from function calls in XMLDoc are saved in a dict 
    structure. The first time a function call is made for a particular instance 
    of XMLDoc will result in a new entry in the internal cache (dict). If the 
    function has an argument, the function entry will get the value equal to a 
    new dict with return values dependent on function argument.
        
    Note: The current version only supports functions with no or one argument.
    """
    
    def __init__(self):
        """ 
        Create internal cache (dict). 
        """
        self.cache={}
        
    def add(self, obj, function, key=None):
        """ 
        Add a function call to cache and save result dependent on the argument 
        key. If key is None, the function has no arguments and the dict entry 
        will simply contain one value which is the return value for the specific 
        function. If key is not none, the value of the dict entry will contain 
        yet another dict with an entry for each argument to the function.
        
        Parameters::
        
            obj --
                The object instance on which the function call is made.
            function --
                The function call for which the result should be saved.
            key --
                Function argument. If None then the function has no arguments.
                Default: None
        """
        # load xmlparser-function
        f = getattr(obj, function)
        # check if there is a key (argument to function f)
        # and get result (call function)
        if key!=None:
            result = f(key, ignore_cache=True)
        else:
            result = f(ignore_cache=True)
        # check if function is already in cache
        if not self.cache.has_key(function):
            # function is not in cache so add both function 
            # and return result which is either a dict or 
            # "normal" value entry dependent on key
            if key!=None:
                self.cache[function] = {key:result}
            else:
                self.cache[function] = result
        else:
            # function is in cache so add result for the 
            # specific argument
            values = self.cache.get(function)
            # ...should not have to do this check, 
            # have we got this far key can not be = None
            # but keep for now
            if key!=None:
                values[key]=result
            else:
                result=values
        return result
                    
    def get(self, obj, function, key=None):
        """ 
        Get the function return value (cached value) for the specific function 
        and key.
        
        Parameters::
        
            obj --
                The object instance on which the function call is made.
            function --
                The function call for which the result is saved.
            key --
                Function argument. If None then the function has no arguments.
                Default: None
                
        Returns::
        
            The return value of the function for the specific key (if any) as 
            saved in the cache.
        """
        # Get function result value/values from cache
        values = self.cache.get(function)
        # check if function could be found in cache
        if values!=None:
            # if key is none then values is = the function return value
            if key is None:
                return values
            # otherwise, use the key (function arg) to get the correct value
            result = values.get(key)
            # check if found in cache
            if result!=None:
                #return result is found
                return result
        # result was not found - add to cache
        return self.add(obj, function, key)

#=======================================================================

class XMLException(Exception):
    """ 
    Class for all XML related errors that can occur in this module. 
    """
    pass
    
