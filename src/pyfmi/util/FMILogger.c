/*
    Copyright (C) 2010 Modelon AB

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as published
    by the Free Software Foundation, or optionally, under the terms of the
    Common Public License version 1.0 as published by IBM.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License, or the Common Public License, for more details.

    You should have received copies of the GNU General Public License
    and the Common Public License along with this program.  If not,
    see <http://www.gnu.org/licenses/> or
    <http://www.ibm.com/developerworks/library/os-cpl.html/> respectively.
*/

/*
 *  FMILogger.c contains structs for conversion of the FMI struct (fmiCallbackFunctions) 
 *  to another struct of similarly type which is imported in the FMI wrapper in
 *  jmodelica.fmi.FMIModel.
 */

#include <stdarg.h>
#include <stdio.h>

//Define the fmi typedefs (See FMI documentation)

typedef void*        fmiComponent;
typedef unsigned int fmiValueReference; 
typedef double       fmiReal   ;
typedef int          fmiInteger;
typedef char         fmiBoolean;
typedef const char*  fmiString ;

typedef enum  {fmiOK,
                  fmiWarning,
                  fmiDiscard,
                  fmiError,
                  fmiFatal} fmiStatus;

//Define the callback funtions

typedef void  (*py_logger_type)(fmiComponent, fmiString, fmiStatus, fmiString, fmiString);
typedef void  (*fmi_logger_type)(fmiComponent, fmiString, fmiStatus, fmiString, fmiString, ...);
typedef void* (*fmi_allocate_type)(size_t, size_t);
typedef void  (*fmi_free_type)(void*);

//Define the two different structs.
//
//py_fmiCallbackFunctions are used by Python
//c_fmiCallbackFunstions are used by C
//
//The only difference between the two are the variable number of inputs (...)
//which ctypes can not handle.

typedef struct py_fmiCallbackFunctions {
    py_logger_type logger;
    fmi_allocate_type allocateMemory;
    fmi_free_type freeMemory;
} py_fmiCallbackFunctions;

typedef struct c_fmiCallbackFunctions {
    fmi_logger_type logger;
    fmi_allocate_type allocateMemory;
    fmi_free_type freeMemory;
} c_fmiCallbackFunctions;

static py_logger_type python_logger_callback;
c_fmiCallbackFunctions c_struct;

#define BUF_SIZE 1000 //Defines the buffer size

void fmuLoggerCallback(fmiComponent c, fmiString instanceName, fmiStatus status, fmiString category, fmiString message, ...) {
    //Transforms the ... into a buffer which is then passed to python as a message
    static char buf[BUF_SIZE];
    va_list va;
    va_start(va, message);
    vsnprintf(buf, BUF_SIZE, message, va);
    va_end(va);
    python_logger_callback(c,instanceName,status,category,buf);
}


c_fmiCallbackFunctions* pythonCallbacks(py_fmiCallbackFunctions py_struct) {
    //Transforms the python struct to the C struct
    python_logger_callback = py_struct.logger;
    c_struct.logger = fmuLoggerCallback;
    c_struct.allocateMemory = py_struct.allocateMemory;
    c_struct.freeMemory = py_struct.freeMemory;
    
    return &c_struct;
}


