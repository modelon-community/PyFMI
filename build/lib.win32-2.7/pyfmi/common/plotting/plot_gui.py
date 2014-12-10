#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2010 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import matplotlib
matplotlib.interactive(True)
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib import rcParams
import fnmatch
import re

#GUI modules
try:
    import wx
    import wx.lib.agw.customtreectrl as wxCustom
    import wx.lib.agw.aui as aui
except ImportError:
    print "WX-Python not found. The GUI will not work."

#JModelica related imports
try:
    from pyfmi.common.io import ResultDymolaTextual
    from pyfmi.common.io import ResultDymolaBinary
    from pyfmi.common.io import ResultCSVTextual
    from pyfmi.common.io import JIOError
except ImportError:
    try:
        from pyjmi.common.io import ResultDymolaTextual
        from pyjmi.common.io import ResultDymolaBinary
        from pyjmi.common.io import ResultCSVTextual
        from pyjmi.common.io import JIOError
    except ImportError:
        print "JModelica Python package was not found."

#Import general modules
import os as O

ID_GRID    = 15001
ID_LICENSE = 15002
ID_LABELS  = 15003
ID_AXIS    = 15004
ID_MOVE    = 15005
ID_ZOOM    = 15006
ID_RESIZE  = 15007
ID_LINES   = 15008
ID_CLEAR   = 15009

def convert_filter(expression):
    """
    Convert a filter based on unix filename pattern matching to a
    list of regular expressions.
    """
    regexp = []
    if isinstance(expression,str):
        regex = fnmatch.translate(expression)
        regexp = [re.compile(regex)]
    elif isinstance(expression,list):
        for i in expression:
            regex = fnmatch.translate(i)
            regexp.append(re.compile(regex))
    else:
        raise Exception("Unknown input.")
    return regexp
    
def match(name, filter_list):
    found = False
    for j in range(len(filter_list)):
        if re.match(filter_list[j], name):
            found = True
            break
    
    return found
    

class MainGUI(wx.Frame):
    sizeHeightDefault=900
    sizeLengthDefault=675
    sizeHeightMin=100
    sizeLengthMin=130
    sizeTreeMin=200
    sizeTreeDefault=sizeTreeMin+40
    
    def __init__(self, parent, ID, filename=None):
        
        self.title = "JModelica.org Plot GUI"
        wx.Frame.__init__(self, parent, ID, self.title,
                         wx.DefaultPosition, wx.Size(self.sizeHeightDefault, self.sizeLengthDefault))
                         
        #Handle idle events
        #wx.IdleEvent.SetMode(wx.IDLE_PROCESS_SPECIFIED)
        
        #Variables for the results
        self.ResultFiles = [] #Contains all the result files
        self.PlotVariables = [[]] #Contains all the variables for the different plots
        self.ResultIndex = 0 #Index of the result file
        self.PlotIndex = 0 #Index of the plot variables connected to the different plots
        
        #Settings variables
        self.grid = True
        self.zoom = True
        self.move = False
        
        #Create menus and status bars
        self.CreateStatusBar() #Create a statusbar at the bottom
        self.CreateMenu() #Create the normal menu
        
        #Create the main window
        self.verticalSplitter = wx.SplitterWindow(self, -1, style = wx.CLIP_CHILDREN | wx.SP_LIVE_UPDATE | wx.SP_3D)
        
        #Create the positioners
        self.leftPanel = wx.Panel(self.verticalSplitter)
        self.leftSizer = wx.BoxSizer(wx.VERTICAL)
        self.rightPanel = wx.Panel(self.verticalSplitter)
        self.rightSizer = wx.BoxSizer(wx.VERTICAL)
        
        #Create the panels (Tree and Plot)
        
        if wx.VERSION < (2,8,11,0):
            self.tree = VariableTree(self.leftPanel,style = wx.SUNKEN_BORDER | wxCustom.TR_HAS_BUTTONS | wxCustom.TR_HAS_VARIABLE_ROW_HEIGHT | wxCustom.TR_HIDE_ROOT | wxCustom.TR_ALIGN_WINDOWS)
            self.noteBook = aui.AuiNotebook(self.rightPanel, style= aui.AUI_NB_TOP | aui.AUI_NB_TAB_SPLIT | aui.AUI_NB_TAB_MOVE | aui.AUI_NB_SCROLL_BUTTONS | aui.AUI_NB_CLOSE_ON_ACTIVE_TAB | aui.AUI_NB_DRAW_DND_TAB)
        else:
            self.tree = VariableTree(self.leftPanel,style = wx.SUNKEN_BORDER, agwStyle = wxCustom.TR_HAS_BUTTONS | wxCustom.TR_HAS_VARIABLE_ROW_HEIGHT | wxCustom.TR_HIDE_ROOT | wxCustom.TR_ALIGN_WINDOWS)
            self.noteBook = aui.AuiNotebook(self.rightPanel, agwStyle= aui.AUI_NB_TOP | aui.AUI_NB_TAB_SPLIT | aui.AUI_NB_TAB_MOVE | aui.AUI_NB_SCROLL_BUTTONS | aui.AUI_NB_CLOSE_ON_ACTIVE_TAB | aui.AUI_NB_DRAW_DND_TAB)
        self.plotPanels = [PlotPanel(self.noteBook,self.grid, move=self.move, zoom=self.zoom)]
        self.noteBook.AddPage(self.plotPanels[0],"Plot 1")
        self.filterPanel = FilterPanel(self.leftPanel, self.tree)
        
        
        #Add the panels to the positioners
        self.leftSizer.Add(self.tree,1,wx.EXPAND)
        self.leftSizer.Add(self.filterPanel,0,wx.EXPAND)
        self.rightSizer.Add(self.noteBook,1,wx.EXPAND)
        

        self.verticalSplitter.SplitVertically(self.leftPanel, self.rightPanel,self.sizeTreeDefault)
        #self.verticalSplitter.SetMinimumPaneSize(self.sizeTreeMin)
        self.verticalSplitter.SetMinimumPaneSize(self.filterPanel.GetBestSize()[0])
        
        
        #Position the main windows
        self.leftPanel.SetSizer(self.leftSizer)
        self.rightPanel.SetSizer(self.rightSizer)
        self.mainSizer = wx.BoxSizer() #Create the main positioner
        self.mainSizer.Add(self.verticalSplitter, 1, wx.EXPAND) #Add the vertical splitter
        self.SetSizer(self.mainSizer) #Set the positioner to the main window
        self.SetMinSize((self.sizeHeightMin,self.sizeLengthMin)) #Set minimum sizes
        
        #Bind the exit event from the "cross"
        self.Bind(wx.EVT_CLOSE, self.OnMenuExit)
        #Bind the tree item checked event
        self.tree.Bind(wxCustom.EVT_TREE_ITEM_CHECKED, self.OnTreeItemChecked)
        #Bind the key press event
        self.tree.Bind(wx.EVT_KEY_DOWN, self.OnKeyPress)
        #Bind the closing of a tab
        self.Bind(aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.OnCloseTab, self.noteBook)
        #Bind the changing of a tab
        self.Bind(aui.EVT_AUINOTEBOOK_PAGE_CHANGING, self.OnTabChanging, self.noteBook)
        #Bind the changed of a tab
        self.Bind(aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.OnTabChanged, self.noteBook)
        if not filename == None:
            self._OpenFile(filename)
        
        self.Centre(True) #Position the GUI in the centre of the screen
        self.Show(True) #Show the Plot GUI
        
    def CreateMenu(self):
        #Creating the menu
        filemenu = wx.Menu()
        helpmenu = wx.Menu()
        editmenu = wx.Menu()
        viewmenu = wx.Menu()
        menuBar  = wx.MenuBar()
        
        #Create the menu options
        # Main
        self.menuOpen  = filemenu.Append(wx.ID_OPEN, "&Open\tCtrl+O","Open a result.")
        self.menuSaveFig = filemenu.Append(wx.ID_SAVE, "&Save\tCtrl+S", "Save the current figure.")
        filemenu.AppendSeparator() #Append a seperator between Open and Exit
        self.menuExit  = filemenu.Append(wx.ID_EXIT,"E&xit\tCtrl+X"," Terminate the program.")
        
        # Edit
        self.editAdd  = editmenu.Append(wx.ID_ADD,"A&dd Plot","Add a plot window.")
        self.editClear = editmenu.Append(wx.ID_CLEAR, "Clear Plot", "Clear the current plot window.")
        editmenu.AppendSeparator()
        self.editAxisLabels = editmenu.Append(ID_AXIS,"Axis / Labels", "Edit the axis and labels of the current plot.")
        self.editLinesLegends = editmenu.Append(ID_LINES, "Lines / Legends", "Edit the lines and the legend of the current plot.")
        
        # View
        self.viewGrid  = viewmenu.Append(ID_GRID,"&Grid","Show/Hide Grid.",kind=wx.ITEM_CHECK)
        viewmenu.AppendSeparator() #Append a seperator
        self.viewMove = viewmenu.Append(ID_MOVE,"Move","Use the mouse to move the plot.",kind=wx.ITEM_RADIO)
        self.viewZoom = viewmenu.Append(ID_ZOOM,"Zoom","Use the mouse for zooming.",kind=wx.ITEM_RADIO)
        viewmenu.AppendSeparator()
        self.viewResize = viewmenu.Append(ID_RESIZE, "Resize", "Resize the current plot.")
        
        #Check items
        viewmenu.Check(ID_GRID, self.grid)
        viewmenu.Check(ID_ZOOM, self.zoom)
        viewmenu.Check(ID_MOVE, self.move)
        
        # Help
        self.helpLicense = helpmenu.Append(ID_LICENSE, "License","Show the license.")
        self.helpAbout = helpmenu.Append(wx.ID_ABOUT, "&About"," Information about this program.")
  
        #Setting up the menu
        menuBar.Append(filemenu,"&File") #Adding the "filemenu" to the MenuBar
        menuBar.Append(editmenu,"&Edit") #Adding the "editmenu" to the MenuBar
        menuBar.Append(viewmenu,"&View") #Adding the "viewmenu" to the MenuBar
        menuBar.Append(helpmenu,"&Help") #Adding the "helpmenu" to the MenuBar
        
        #Binding the events
        self.Bind(wx.EVT_MENU, self.OnMenuOpen,    self.menuOpen)
        self.Bind(wx.EVT_MENU, self.OnMenuSaveFig, self.menuSaveFig)
        self.Bind(wx.EVT_MENU, self.OnMenuExit,    self.menuExit)
        self.Bind(wx.EVT_MENU, self.OnMenuAdd,     self.editAdd)
        self.Bind(wx.EVT_MENU, self.OnMenuClear,     self.editClear)
        self.Bind(wx.EVT_MENU, self.OnMenuAxisLabels,    self.editAxisLabels)
        self.Bind(wx.EVT_MENU, self.OnMenuLinesLegends,  self.editLinesLegends)
        self.Bind(wx.EVT_MENU, self.OnMenuResize,  self.viewResize)
        self.Bind(wx.EVT_MENU, self.OnMenuGrid,    self.viewGrid)
        self.Bind(wx.EVT_MENU, self.OnMenuMove,    self.viewMove)
        self.Bind(wx.EVT_MENU, self.OnMenuZoom,    self.viewZoom)
        self.Bind(wx.EVT_MENU, self.OnMenuLicense, self.helpLicense)
        self.Bind(wx.EVT_MENU, self.OnMenuAbout,   self.helpAbout)
                
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        
        #Set keyboard shortcuts       
        hotKeysTable = wx.AcceleratorTable([(wx.ACCEL_CTRL, ord("O"), self.menuOpen.GetId()),
                                            (wx.ACCEL_CTRL, ord("S"), self.menuSaveFig.GetId()),
                                            (wx.ACCEL_CTRL, ord("X"), self.menuExit.GetId())])
        self.SetAcceleratorTable(hotKeysTable)
        
        #Disable Lines and Legends
        self.editLinesLegends.Enable(False)
    
    def OnMenuMove(self, event):
        self.move = True
        self.zoom = False
        
        for i in range(self.noteBook.GetPageCount()):
            self.noteBook.GetPage(i).UpdateSettings(move = self.move,
                                                    zoom = self.zoom)
        
    def OnMenuZoom(self, event):
        self.move = False
        self.zoom = True
        
        for i in range(self.noteBook.GetPageCount()):
            self.noteBook.GetPage(i).UpdateSettings(move = self.move,
                                                    zoom = self.zoom)
    
    def OnMenuExit(self, event):
        self.Destroy() #Close the GUI
    
    def OnMenuAbout(self, event):
        dlg = wx.MessageDialog(self, 'JModelica.org Plot GUI.\n', 'About',
                                        wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
        
    def OnMenuResize(self, event):
        
        IDPlot = self.noteBook.GetSelection()
        self.noteBook.GetPage(IDPlot).ReSize()
        
    def OnMenuOpen(self, event):
        #Open the file window
        dlg = wx.FileDialog(self, "Open result file(s)",wildcard="Text files (.txt)|*.txt|MATLAB files (.mat)|*.mat|All files (*.*)|*.*",
                            style=wx.FD_MULTIPLE)
        
        #If OK load the results
        if dlg.ShowModal() == wx.ID_OK:
            for n in dlg.GetFilenames():
                self._OpenFile(O.path.join(dlg.GetDirectory(),n))
        
        dlg.Destroy() #Destroy the popup window
    
    def OnMenuSaveFig(self, event):
        #Open the file window
        dlg = wx.FileDialog(self, "Choose a filename to save to",wildcard="Portable Network Graphics (*.png)|*.png|" \
                                                                          "Encapsulated Postscript (*.eps)|*.eps|" \
                                                                          "Enhanced Metafile (*.emf)|*.emf|" \
                                                                          "Portable Document Format (*.pdf)|*.pdf|" \
                                                                          "Postscript (*.ps)|*.ps|" \
                                                                          "Raw RGBA bitmap (*.raw *.rgba)|*.raw;*.rgba|" \
                                                                          "Scalable Vector Graphics (*.svg *.svgz)|*.svg;*.svgz",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        
        #If OK save the figure
        if dlg.ShowModal() == wx.ID_OK:
            self.SetStatusText("Saving figure...") #Change the statusbar

            IDPlot = self.noteBook.GetSelection()
            self.noteBook.GetPage(IDPlot).Save(dlg.GetPath())
            
            self.SetStatusText("") #Change the statusbar
        
        dlg.Destroy() #Destroy the popup window
        
    
    def OnMenuAdd(self, event):
        #Add a new list for the plot variables connect to the plot
        self.PlotVariables.append([])
        self.PlotIndex += 1
        
        #Add a new plot panel to the notebook
        self.plotPanels.append(PlotPanel(self.noteBook,self.grid,move=self.move, zoom=self.zoom))
        self.noteBook.AddPage(self.plotPanels[-1],"Plot "+str(self.PlotIndex+1))
        
        #Enable labels and axis options
        self.editAxisLabels.Enable(True)
    
    def OnMenuClear(self, event):
        #Clear the current activated plot window
        
        IDPlot = self.noteBook.GetSelection()
        
        if IDPlot != -1:
            plotWindow = self.noteBook.GetPage(IDPlot)
            
            #Uncheck all variables
            for i,var in enumerate(self.noteBook.GetPage(IDPlot).GetPlotVariables()):
                self.tree.CheckItem2(var[1],checked=False,torefresh=True)
            
            #Delete all variables
            plotWindow.DeleteAllPlotVariables()
            
            #Disable Lines and Legends
            self.editLinesLegends.Enable(False)
            
            plotWindow.SetDefaultSettings()
            plotWindow.UpdateSettings(axes=[0.0,1.0,0.0,1.0])
            plotWindow.Draw()
            plotWindow.UpdateSettings(axes=[None,None,None,None])
    
    def OnMenuLinesLegends(self, event):
        IDPlot = self.noteBook.GetSelection()
        plotWindow = self.noteBook.GetPage(IDPlot)
        
        #Create the axis dialog
        dlg = DialogLinesLegends(self,self.noteBook.GetPage(IDPlot))
        
        #Open the dialog and update options if OK
        if dlg.ShowModal() == wx.ID_OK:
            dlg.ApplyChanges() #Apply Changes
            
            legend = dlg.GetValues()

            plotWindow.UpdateSettings(legendposition=legend)
            plotWindow.Draw()
            
        #Destroy the dialog
        dlg.Destroy()
        
    
    def OnMenuAxisLabels(self, event):
        IDPlot = self.noteBook.GetSelection()
        plotWindow = self.noteBook.GetPage(IDPlot)
        
        #Create the axis dialog
        dlg = DialogAxisLabels(self,self.noteBook.GetPage(IDPlot))
        
        #Open the dialog and update options if OK
        if dlg.ShowModal() == wx.ID_OK:
            
            xmax,xmin,ymax,ymin,title,xlabel,ylabel,xscale,yscale = dlg.GetValues()
            
            try:
                xmax=float(xmax)
            except ValueError:
                xmax=None
            try:
                xmin=float(xmin)
            except ValueError:
                xmin=None
            try:
                ymax=float(ymax)
            except ValueError:
                ymax=None
            try:
                ymin=float(ymin)
            except ValueError:
                ymin=None
            
            plotWindow.UpdateSettings(axes=[xmin,xmax,ymin,ymax],
                                    title=title,xlabel=xlabel,ylabel=ylabel,
                                    xscale=xscale, yscale=yscale)
            plotWindow.DrawSettings()
        
        #Destroy the dialog
        dlg.Destroy()
        
    def OnMenuGrid(self, event):
        self.grid = not self.grid
        
        for i in range(self.noteBook.GetPageCount()):
            self.noteBook.GetPage(i).UpdateSettings(grid = self.grid)
            self.noteBook.GetPage(i).DrawSettings()
        
    def OnTreeItemChecked(self, event):
        self.SetStatusText("Drawing figure...")
        
        item = event.GetItem()
        
        #ID = self.tree.FindIndexParent(item)
        ID = -1 #Not used
        IDPlot = self.noteBook.GetSelection()
        
        if IDPlot != -1: #If there exist a plot window
            
            data = self.tree.GetPyData(item)
            
            #Store plot variables or "unstore"
            if self.tree.IsItemChecked(item): #Draw
                
                #Add to Plot panel
                self.noteBook.GetPage(IDPlot).AddPlotVariable(ID,item,data)
                
            else: #Undraw
            
                #Remove from panel
                #self.noteBook.GetPage(IDPlot).DeletePlotVariable(item)
                self.noteBook.GetPage(IDPlot).DeletePlotVariable(data["variable_id"])
                
            self.noteBook.GetPage(IDPlot).Draw()
            
            lines = self.noteBook.GetPage(IDPlot).GetLines()
            if len(lines) != 0:
                #Enable Lines and Legends
                self.editLinesLegends.Enable(True)
            else:
                #Disable Lines and Legends
                self.editLinesLegends.Enable(False)
        
        else: #Dont allow an item to be checked if there exist no plot window
            self.tree.CheckItem2(item,checked=False,torefresh=True)
        
        self.SetStatusText("")
        
    def OnKeyPress(self, event):
        keycode = event.GetKeyCode() #Get the key pressed
        
        #If the key is Delete
        if keycode == wx.WXK_DELETE:
            self.SetStatusText("Deleting Result...")

            ID = self.tree.FindIndexParent(self.tree.GetSelection())
            data = self.tree.GetPyData(self.tree.GetSelection())
            IDPlot = self.noteBook.GetSelection()

            if ID >= 0: #If id is less then 0, no item is selected
                
                self.ResultFiles.pop(ID) #Delete the result object from the list
                self.tree.DeleteParent(self.tree.GetSelection())
                
                #Redraw
                for i in range(self.noteBook.GetPageCount()):
                    #self.noteBook.GetPage(i).DeletePlotVariable(ID=ID)
                    self.noteBook.GetPage(i).DeletePlotVariable(global_id=data["result_id"])
                    self.noteBook.GetPage(i).Draw()

            self.SetStatusText("")
    
    def OnCloseTab(self, event):
        self.OnTabChanging(event)
        self.PlotVariables.pop(event.GetSelection()) #Delete the plot
        self.plotPanels.pop(event.GetSelection()) #MAYBE!
                            #variables associated with the current plot
        
        #Disable changing of labels and axis if there is no Plot
        if self.noteBook.GetPageCount() == 1:
            self.editAxisLabels.Enable(False)
            self.editLinesLegends.Enable(False)
                            
    def OnTabChanging(self, event):
        IDPlot = self.noteBook.GetSelection()
        
        #Uncheck the items related to the previous plot
        if IDPlot != -1:
            for i,var in enumerate(self.noteBook.GetPage(IDPlot).GetPlotVariables()):
                self.tree.CheckItem2(var[1],checked=False,torefresh=True)
            
            lines = self.noteBook.GetPage(IDPlot).GetLines()
            if len(lines) != 0:
                #Enable Lines and Legends
                self.editLinesLegends.Enable(True)
            else:
                #Disable Lines and Legends
                self.editLinesLegends.Enable(False)
    
    def OnTabChanged(self,event):
        IDPlot = self.noteBook.GetSelection()
        
        #Check the items related to the previous plot
        if IDPlot != -1:
            for i,var in enumerate(self.noteBook.GetPage(IDPlot).GetPlotVariables()):
                self.tree.CheckItem2(var[1],checked=True,torefresh=True)
            
            lines = self.noteBook.GetPage(IDPlot).GetLines()
            if len(lines) != 0:
                #Enable Lines and Legends
                self.editLinesLegends.Enable(True)
            else:
                #Disable Lines and Legends
                self.editLinesLegends.Enable(False)

    def OnMenuLicense(self, event):
        
        desc = "Copyright (C) 2011 Modelon AB\n\n"\
"This program is free software: you can redistribute it and/or modify "\
"it under the terms of the GNU General Public License as published by "\
"the Free Software Foundation, version 3 of the License.\n\n"\
"This program is distributed in the hope that it will be useful, "\
"but WITHOUT ANY WARRANTY; without even the implied warranty of "\
"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "\
"GNU General Public License for more details.\n\n"\
"You should have received a copy of the GNU General Public License "\
"along with this program.  If not, see <http://www.gnu.org/licenses/>. "
        
        dlg = wx.MessageDialog(self, desc, 'License', wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def _OpenFile(self,filename):
        
        # Extract filename and result name
        n = str(filename)
        if n.find('\\') > n.find('/'):
            res_name = n.split('\\')[-1]
        else:
            res_name = n.split('/')[-1]
        failToLoad = False
        self.SetStatusText("Loading "+n+"...") #Change the statusbar
        
        #Find out if the result is a textual or binary file
        if n.lower().endswith(".txt"): #Textual file
            try:
                self.ResultFiles.append((res_name,ResultDymolaTextual(n)))
            except (JIOError, IOError):
                self.SetStatusText("Could not load "+n+".") #Change the statusbar
                failToLoad = True
                
        elif n.lower().endswith(".mat"): #Binary file
            try:
                self.ResultFiles.append((res_name,ResultDymolaBinary(n)))
            except (TypeError, IOError):
                self.SetStatusText("Could not load "+n+".") #Change the statusbar
                failToLoad = True
                
        elif n.lower().endswith(".csv"): #Binary file
            try:
                self.ResultFiles.append((res_name,ResultCSVTextual(n)))
            except (TypeError, IOError):
                self.SetStatusText("Could not load "+n+".") #Change the statusbar
                failToLoad = True
                
        else:
            self.SetStatusText("Could not load "+n+".") #Change the statusbar
            failToLoad = True
            
        if failToLoad:
            self.SetStatusText("Could not open file '" + n + "'!\n")
        else:
        
            self.SetStatusText("Populating tree for " +n+"...")
            
            self.tree.AddTreeNode(self.ResultFiles[-1][1], self.ResultFiles[-1][0], 
                                        self.filterPanel.checkBoxTimeVarying.GetValue(),
                                        self.filterPanel.checkBoxParametersConstants.GetValue(),
                                        self.filterPanel.GetFilter())
            
            self.ResultIndex += 1 #Increment the index
            
            self.SetStatusText("") #Change the statusbar
                
class VariableTree(wxCustom.CustomTreeCtrl):
    def __init__(self, *args, **kwargs):
        super(VariableTree, self).__init__(*args, **kwargs)
        
        #Add the root item
        self.root = self.AddRoot("Result(s)")
        #Root have children
        self.SetItemHasChildren(self.root)
        #Internal counter for all children
        self.global_id = 0 #Global ID for each loaded results (unique for each results
        self.local_id  = 0 #Local ID for each loaded variable (unique for each variable and results)
        self.node_id   = 0 #Node ID for each node with childrens
        
        #List of hidden children
        self.hidden_children = []
        self.hidden_nodes    = {}
        self.nodes           = {}
        
        #Internal flags
        self._update_top_siblings = True
        
    
    def RefreshSelectedUnder(self, item):
        """
        Refreshes the selected items under the given item.

        :param `item`: an instance of L{GenericTreeItem}.
        """
        if self._freezeCount:
            return

        if item.IsSelected():
            self.RefreshLine(item)

        children = item.GetChildren()
        for child in children:
            if child.IsSelected():
                self.RefreshLine(child)
    
    def AddTreeNode(self, resultObject, name,timeVarying=None,parametersConstants=None,filter=None):
        #Freeze the window temporarely
        self.Freeze()
        
        #Add a new dictionary for the nodes
        self.nodes[self.global_id] = {}
        self._update_top_siblings = True
        
        child = self.AppendItem(self.root, name, data={"result_id":self.global_id, "node_id": self.node_id})
        self.SetItemHasChildren(child,True)
        
        self.nodes[self.global_id][self.node_id] = {"node":child, "node_id":self.node_id, "name":name, "parent_node":self.root, "parent_node_id": -1}
        self.nodes[self.global_id][-1] = {"node":child, "node_id":self.node_id, "name":name, "parent_node":self.root, "parent_node_id": -2}
        self.node_id = self.node_id + 1 #Increment the nodes
        
        rec = {"root":child}
        
        for item in resultObject.name:
            spl = item.split(".")
            
            #Python object for storing data related to the variable
            data={}
            data["timevarying"] = None #resultObject.is_variable(item)
            #data["traj"] = resultObject.get_variable_data(item)
            data["traj"] = resultObject
            data["name"] = item
            data["full_name"] = item
            data["result_id"] = self.global_id
            data["variable_id"] = self.local_id = self.local_id + 1
            data["result_object"] = resultObject

            if len(spl)==1:
                data["parents"] = child
                data["child"]   = item
                data["node_id"] = self.GetPyData(child)["node_id"]
                self.AppendItem(child, item,ct_type=1, data=data)
                
            else:
                #Handle variables of type der(---.---.x)
                if spl[0].startswith("der(") and spl[-1].endswith(")"):
                    spl[0]=spl[0][4:]
                    spl[-1] = "der("+spl[-1]
                
                tmp_str = ""
                tmp_str_old = ""      
                for i in range(len(spl)-1):
                    #See if the sub directory already been added, else add
                    tmp_str_old = tmp_str
                    tmp_str += spl[i]
                    try:
                        rec[tmp_str]
                    except KeyError:
                        local_data = {"result_id":self.global_id, "node_id":self.node_id}
                        if i==0:
                            rec[tmp_str] = self.AppendItem(child, spl[i], data=local_data)
                            local_dict = {"node":rec[tmp_str], "node_id":self.node_id, "name":spl[i], "parent_node":child, "parent_node_id": -1}
                            self.nodes[self.global_id][self.node_id] = local_dict
                        else:
                            rec[tmp_str] = self.AppendItem(rec[tmp_str_old], spl[i], data=local_data)
                            local_dict = {"node":rec[tmp_str], "node_id":self.node_id, "name":spl[i], "parent_node":rec[tmp_str_old], "parent_node_id": self.GetPyData(rec[tmp_str_old])["node_id"]}
                            self.nodes[self.global_id][self.node_id] = local_dict
                        self.SetItemHasChildren(rec[tmp_str],True)
                        
                        self.node_id = self.node_id + 1 #Increment the nodes
                else:
                    data["parents"] = rec[tmp_str]
                    data["child"]   = spl[-1]
                    data["node_id"] = self.GetPyData(rec[tmp_str],)["node_id"]
                    self.AppendItem(rec[tmp_str], spl[-1], ct_type=1, data=data)
                    
        self.SortChildren(child)
        
        #Increment global id
        self.global_id = self.global_id + 1
        
        #print "Adding: ", name, "Options: ", timeVarying, parametersConstants, filter
        #print "Condition: ", timeVarying == False or parametersConstants == False or filter != None
        
        #Hide nodes if options are choosen
        if timeVarying == False or parametersConstants == False or filter != None:
            self.HideNodes(timeVarying,parametersConstants,filter)
        
        #Un-Freeze the window
        self.Thaw()
    
    def FindLoneChildDown(self, child):
        """
        Search for the youngest child down the tree from "child".
        
        Parameters::
        
            child - The item from where the search should start.
            
        Returns::
        
            child - The youngest child from the starting point.
        """
        while True:
            nextItem,cookie = self.GetNextChild(child,0)
            if nextItem != None:
                child = nextItem
            else:
                break
        return child
    
    def FindFirstSiblingUp(self,child,itemParent):
        """
        Search for the first sibling of "child" going up in tree.
        """
        while child != itemParent:
            nextItem = self.GetNextSibling(child)
            
            if nextItem != None:
                return nextItem

            child = self.GetItemParent(child)
        return child
    
    def HideNodes(self, showTimeVarying=None, showParametersConstants=None, filter=None):
        """
        Hide nodes depending on the input.
        
        Parameters::
        
            showTimeVarying - Hides or Shows the time varying variables.
            
            showParametersConstants - Hides or Show the parameters.
        """
        itemParent = self.GetRootItem()
        child,cookie = self.GetFirstChild(itemParent)
        found_child = child
        
        top_siblings = self.FindTopSiblings()
        
        #Hide items if any of the options are True
        if showTimeVarying == False or showParametersConstants == False or filter != None:
            while child != itemParent and child != None:
                already_hidden = False
                
                #Find the first youngest child
                found_child = self.FindLoneChildDown(child)
                
                #Find the first sibling up
                child = self.FindFirstSiblingUp(found_child, itemParent)
                data  = self.GetPyData(found_child)
                
                if found_child in top_siblings:
                    print "Found child in top siblings, ", self.GetItemText(found_child)
                    continue
                
                #print "Found child:", self.GetItemText(found_child)
                #print "Child: ", self.GetItemText(child), self.GetPyData(child), "Has Children: ", self.HasChildren(child)
                
                if data == None:
                    print "Found (wrong) child:", self.GetItemText(found_child)
                    raise Exception
                
                try:
                    data["timevarying"]
                except KeyError:
                    print "Found (wrong (exception)) child:", self.GetItemText(found_child)
                    raise Exception
                    
                if data["timevarying"] == None:
                    data["timevarying"] = data["result_object"].is_variable(data["full_name"])
                
                #Enable or disable depending on input to method
                if showTimeVarying == False and data["timevarying"]:
                    self.HideItem(found_child, showTimeVarying)
                    
                    #Delete the parent if it has no children
                    self.HideNodeItem(found_child)
                    
                    already_hidden = True
                    
                if showParametersConstants == False and not data["timevarying"]:
                    self.HideItem(found_child, showParametersConstants)
                    
                    #Delete the parent if it has no children
                    self.HideNodeItem(found_child)
                    
                    already_hidden = True
                
                if not already_hidden and filter != None and not match(data["full_name"], filter):
                    self.HideItem(found_child, show=False)
                    
                    #Delete the parent if it has no children
                    self.HideNodeItem(found_child)
        
        #Re-add items if any of the options are True
        if showTimeVarying == True or showParametersConstants == True or filter != None:
            self.AddHiddenItems(showTimeVarying, showParametersConstants, filter)
    
    def FindTopSiblings(self):
        """
        Finds all the siblings one level down from root.
        """
        if self._update_top_siblings:
            itemParent = self.GetRootItem()
            child,cookie = self.GetFirstChild(itemParent)
            
            siblings = []
            while child != None:
                siblings.append(child)
                child = self.GetNextSibling(child)
            self._top_siblings = siblings
        else:
            siblings = self._top_siblings
        self._update_top_siblings = False
            
        return siblings
    
    def AddHiddenItems(self, showTimeVarying=None, showParametersConstants=None, filter=None):

        #print "Adding hidden items: ", showTimeVarying, showParametersConstants, filter
        
        i = 0
        while i < len(self.hidden_children):
            data = self.hidden_children[i]
            matching = False
            
            #Do not add any items!
            if data["timevarying"] and showTimeVarying == False or not data["timevarying"] and showParametersConstants == False:
                i = i+1
                continue
            
            if filter != None:
                matching = match(data["full_name"], filter)
            
            if     data["timevarying"] and showTimeVarying == True and (filter == None or filter != None and matching == True) or \
               not data["timevarying"] and showParametersConstants == True and (filter == None or filter != None and matching == True):
               #or filter != None and match(data["full_name"], filter):
                
                if self.nodes[data["result_id"]][data["node_id"]]["node"] == None:
                    self.AddHiddenNodes(data)
                
                #print "Adding: ", data
                #print "At node: ", self.nodes[data["result_id"]][data["node_id"]]
                item = self.AppendItem(self.nodes[data["result_id"]][data["node_id"]]["node"], data["child"],ct_type=1, data=data)
                if item == None:
                    raise Exception("Something went wrong when adding the variable.")
                
                self.hidden_children.pop(i)
                i = i-1
            i = i+1
    
    def AddHiddenNodes(self, data):
        
        node = self.nodes[data["result_id"]][data["node_id"]]
        nodes_to_be_added = [node]
        
        while node["node"] == None and node["parent_node_id"] != -1:
            node = self.nodes[data["result_id"]][node["parent_node_id"]]
            
            if node["node"] != None:
                break
            
            nodes_to_be_added.append(node)
        
        #print "Nodes to be added: ", nodes_to_be_added
        
        for i in range(len(nodes_to_be_added)):
            node = nodes_to_be_added[-(i+1)]
            
            #print "Adding node: ", node, " at ", self.nodes[data["result_id"]][node["parent_node_id"]], " or ", self.nodes[data["result_id"]][-1], data
            local_data = {"result_id":data["result_id"], "node_id":node["node_id"]}
            """
            if node["parent_node_id"] == -1:
                item = self.AppendItem(self.nodes[data["result_id"]][-1], node["name"], data=local_data)
            else:
                item = self.AppendItem(node["parent_node_id"], node["name"], data=local_data)
            """
            item = self.AppendItem(self.nodes[data["result_id"]][node["parent_node_id"]]["node"], node["name"], data=local_data)
            #item = self.AppendItem(node["parent_node"], node["name"], data=local_data)
            self.SetItemHasChildren(item, True)
            
            self.nodes[data["result_id"]][node["node_id"]]["node"] = item
            
            #print "Node info after adding: ", self.nodes[data["result_id"]][node["node_id"]]
            
                    
    def HideNodeItem(self, item):
        """
        Deletes the parents that does not have any children
        """
        parent = self.GetItemParent(item)
        top_siblings = self.FindTopSiblings()
        
        while self.HasChildren(parent) == False and parent not in top_siblings:
            old_parent = self.GetItemParent(parent)
            
            #Add the deleted nodes to the hidden list so that we can recreate the list
            #self.hidden_nodes.append(self.GetPyData(parent))
            #self.hidden_nodes[self.GetPyData(parent)["node_id"]] = [self.GetPyData(parent), old_parent]
            #self.nodes[self.GetPyData(parent)["result_id"]][self.GetPyData(parent)["node_id"]][0] = None
            self.nodes[self.GetPyData(parent)["result_id"]][self.GetPyData(parent)["node_id"]]["node"] = None
            
            self.Delete(parent)
            parent = old_parent
    
    def HideItem(self, item, show):
        data = self.GetPyData(item)
        
        if not show:
            self.hidden_children.append(data)
            self.Delete(item)
    
    def DeleteParent(self, item):
        """
        Delete the oldest parent of item, except root.
        """
        
        if item == self.GetRootItem():
            return False
        
        parentItem = self.GetItemParent(item)
        
        while parentItem != self.GetRootItem():
            item = parentItem
            parentItem = self.GetItemParent(item)
        
        #Remove also the hidden items contained in the hidden children list
        data = self.GetPyData(item)
        i = 0
        while i < len(self.hidden_children):
            if self.hidden_children[i]["result_id"] == data["result_id"]:
                self.hidden_children.pop(i)
                i = i-1
            i = i+1
            
        #Delete hidden nodes
        self.nodes.pop(data["result_id"])
        
        self.Delete(item) #Delete the parent from the Tree
        
    def FindIndexParent(self, item):
        """
        Find the index of the oldest parent of item from one level down
        from root.
        """

        if item == self.GetRootItem():
            return -1
        
        parentItem = item
        item = self.GetItemParent(parentItem)
        
        while item != self.GetRootItem():
            parentItem = item
            item = self.GetItemParent(parentItem)
        
        root = self.GetRootItem()
        sibling,cookie = self.GetFirstChild(root)
        
        index = 0
        while parentItem != sibling:
            sibling = self.GetNextSibling(sibling)
            index += 1
            
        return index

class DialogLinesLegends(wx.Dialog):
    def __init__(self, parent, plotPage):
        wx.Dialog.__init__(self, parent, -1, "Lines and Legends")
        
        #Get the variables
        self.variables = plotPage.GetPlotVariables()
        #Get the settings
        settings = plotPage.GetSettings()
        #Get the lines
        lines = plotPage.GetLines()
        #First line
        line1 = lines[0]
        
        names = [i[2]["name"] for i in self.variables]
        lineStyles = ["-","--","-.",":"]
        colors = ["Auto","Blue","Green","Red","Cyan","Magenta","Yellow","Black","White"]
        lineStylesNames = ["Solid","Dashed","Dash Dot","Dotted"]
        markerStyles = ["None",'D','s','_','^','d','h','+','*',',','o','.','p','H','v','x','>','<']
        markerStylesNames = ["None","Diamond","Square","Horizontal Line","Triangle Up","Thin Diamond","Hexagon 1","Plus","Star","Pixel","Circle",
                            "Point","Pentagon","Hexagon 2", "Triangle Down", "X", "Triangle Right", "Triangle Left"]
  
        legendPositions = ['Hide','Best','Upper Right','Upper Left','Lower Left','Lower Right','Right','Center Left','Center Right','Lower Center','Upper Center','Center']
        
        self.lineStyles = lineStyles
        self.markerStyles = markerStyles
        self.colors = colors
        
        #Create the legend dict from where to look up positions
        self.LegendDict = dict((item,i) for i,item in enumerate(legendPositions[1:]))
        self.LegendDict["Hide"] = -1
        
        #Create the line style dict
        self.LineStyleDict = dict((item,i) for i,item in enumerate(lineStyles))
        
        #Create the marker dict
        self.MarkerStyleDict = dict((item,i) for i,item in enumerate(markerStyles))
        
        #Create the color dict
        self.ColorsDict = dict((item,i) for i,item in enumerate(colors))
        
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        bagSizer = wx.GridBagSizer(11, 11)
        
        plotLabelStatic  = wx.StaticText(self, -1, "Label")
        plotStyleStatic  = wx.StaticText(self, -1, "Style")
        plotMarkerStyleStatic  = wx.StaticText(self, -1, "Style")
        plotLineStatic   = wx.StaticText(self, -1, "Line")
        plotMarkerStatic = wx.StaticText(self, -1, "Marker")
        plotLegendStatic = wx.StaticText(self, -1, "Legend")
        plotPositionStatic = wx.StaticText(self, -1, "Position")
        plotWidthStatic = wx.StaticText(self, -1, "Width")
        plotColorStatic = wx.StaticText(self, -1, "Color")
        plotMarkerSizeStatic = wx.StaticText(self, -1, "Size")
        
        sizeWidth = 170
        
        #Set the first line as default
        self.plotLines = wx.ComboBox(self, -1, size=(220, -1), choices=names, style=wx.CB_READONLY)
        self.plotLines.SetSelection(0)
        
        #Set the first line as default
        self.plotLineStyle = wx.ComboBox(self, -1, size=(sizeWidth, -1), choices=lineStylesNames, style=wx.CB_READONLY)
        self.plotMarkerStyle = wx.ComboBox(self, -1, size=(sizeWidth, -1), choices=markerStylesNames, style=wx.CB_READONLY)
        
        #Set the first label as default
        self.plotLineName = wx.TextCtrl(self, -1, "", style = wx.TE_LEFT , size =(sizeWidth,-1))
        self.plotWidth = wx.TextCtrl(self, -1, "", style = wx.TE_LEFT, size=(sizeWidth,-1))
        self.plotMarkerSize = wx.TextCtrl(self, -1, "", style = wx.TE_LEFT, size=(sizeWidth,-1))
        self.plotColor = wx.ComboBox(self, -1, choices=colors, size=(sizeWidth,-1),style=wx.CB_READONLY)
        
        #Define the legend
        self.plotLegend = wx.ComboBox(self, -1, size=(sizeWidth, -1), choices=legendPositions, style=wx.CB_READONLY)
        self.plotLegend.SetSelection(plotPage.GetLegendLocation()+1)
        
        #Get the FONT
        font = plotLineStatic.GetFont()
        font.SetWeight(wx.BOLD)
        
        #Set the bold font to the sections
        plotLineStatic.SetFont(font)
        plotMarkerStatic.SetFont(font)
        plotLegendStatic.SetFont(font)
        
        bagSizer.Add(self.plotLines,(0,0),(1,2))
        bagSizer.Add(plotLabelStatic,(1,0))
        bagSizer.Add(self.plotLineName,(1,1))
        bagSizer.Add(plotLineStatic,(2,0),(1,1))
        bagSizer.Add(plotStyleStatic,(3,0))
        bagSizer.Add(self.plotLineStyle,(3,1))
        bagSizer.Add(plotWidthStatic,(4,0))
        bagSizer.Add(self.plotWidth,(4,1))
        bagSizer.Add(plotColorStatic,(5,0))
        bagSizer.Add(self.plotColor,(5,1))
        
        bagSizer.Add(plotMarkerStatic,(6,0),(1,1))
        bagSizer.Add(plotMarkerStyleStatic,(7,0))
        bagSizer.Add(self.plotMarkerStyle,(7,1))
        bagSizer.Add(plotMarkerSizeStatic,(8,0))
        bagSizer.Add(self.plotMarkerSize,(8,1))
        
        
        bagSizer.Add(plotLegendStatic,(9,0),(1,1))
        bagSizer.Add(plotPositionStatic,(10,0))
        bagSizer.Add(self.plotLegend,(10,1))
        
        #Create OK,Cancel and Apply buttons
        self.buttonOK = wx.Button(self, wx.ID_OK)
        self.buttonCancel = wx.Button(self, wx.ID_CANCEL)
        self.buttonApply = wx.Button(self, wx.ID_APPLY)
        
        buttonSizer = wx.StdDialogButtonSizer()
        buttonSizer.AddButton(self.buttonOK)
        buttonSizer.AddButton(self.buttonCancel)
        buttonSizer.AddButton(self.buttonApply)
        buttonSizer.Realize()
        
        #Add information to the sizers
        mainSizer.Add(bagSizer,0,wx.ALL|wx.EXPAND,20)
        mainSizer.Add(buttonSizer,1,wx.ALL|wx.EXPAND,10)
 
        #Set the main sizer to the panel
        self.SetSizer(mainSizer)

        #Set size        
        mainSizer.Fit(self)
        
        #Set the first line as default
        self.ChangeLine(self.variables[0])
        
        #Bind events
        self.Bind(wx.EVT_COMBOBOX, self.OnLineChange)
        self.buttonApply.Bind(wx.EVT_BUTTON, self.OnApply)
    
    def OnApply(self, event):
        self.ApplyChanges()
    
    def OnLineChange(self, event):
        
        if self.plotLines.FindFocus() == self.plotLines:
            ID = self.plotLines.GetSelection()
            self.ChangeLine(self.variables[ID])
    
    def ApplyChanges(self):
        
        ID = self.plotLines.GetSelection()
        
        self.variables[ID][3].name = self.plotLineName.GetValue()
        self.variables[ID][3].style = self.lineStyles[self.plotLineStyle.GetSelection()]
        self.variables[ID][3].width = float(self.plotWidth.GetValue())
        self.variables[ID][3].color = None if self.plotColor.GetSelection()==0 else self.colors[self.plotColor.GetSelection()].lower()
        self.variables[ID][3].marker = self.markerStyles[self.plotMarkerStyle.GetSelection()]
        self.variables[ID][3].markersize = float(self.plotMarkerSize.GetValue())
    
    def ChangeLine(self, var):
        
        self.plotLineStyle.SetSelection(self.LineStyleDict[var[3].style])
        self.plotMarkerStyle.SetSelection(self.MarkerStyleDict[var[3].marker])
        self.plotLineName.SetValue(var[3].name)
        self.plotWidth.SetValue(str(var[3].width))
        self.plotMarkerSize.SetValue(str(var[3].markersize))
        
        if var[3].color == None:
            self.plotColor.SetSelection(0)
        else:
            self.plotColor.SetSelection(self.ColorsDict[var[3].color[0].upper()+var[3].color[1:]])
        
    def GetValues(self):
        return self.LegendDict[self.plotLegend.GetValue()]
    
        
class DialogAxisLabels(wx.Dialog):
    def __init__(self, parent, plotPage):
        wx.Dialog.__init__(self, parent, -1, "Axis and Labels")
        
        settings = plotPage.GetSettings()
        
        plotXAxisStatic = wx.StaticText(self, -1, "X-Axis")
        plotYAxisStatic = wx.StaticText(self, -1, "Y-Axis")
        plotXMaxStatic = wx.StaticText(self, -1, "Max",size =(50,-1))
        plotXMinStatic = wx.StaticText(self, -1, "Min",size =(50,-1))
        plotTitleStatic = wx.StaticText(self, -1, "Title")
        plotXLabelStatic = wx.StaticText(self, -1, "Label")
        plotXScaleStatic = wx.StaticText(self, -1, "Scale")
        plotYMaxStatic = wx.StaticText(self, -1, "Max",size =(50,-1))
        plotYMinStatic = wx.StaticText(self, -1, "Min",size =(50,-1))
        plotYLabelStatic = wx.StaticText(self, -1, "Label")
        plotYScaleStatic = wx.StaticText(self, -1, "Scale")
        
        
        font = plotXAxisStatic.GetFont()
        font.SetWeight(wx.BOLD)
        
        plotXAxisStatic.SetFont(font)
        plotYAxisStatic.SetFont(font)
        
        self.plotYAxisMin = wx.TextCtrl(self, -1, "" if settings["YAxisMin"]==None else str(settings["YAxisMin"]), style = wx.TE_LEFT , size =(150,-1))
        self.plotYAxisMax = wx.TextCtrl(self, -1, "" if settings["YAxisMax"]==None else str(settings["YAxisMax"]), style = wx.TE_LEFT , size =(150,-1))
        self.plotXAxisMin = wx.TextCtrl(self, -1, "" if settings["XAxisMin"]==None else str(settings["XAxisMin"]), style = wx.TE_LEFT , size =(150,-1))
        self.plotXAxisMax = wx.TextCtrl(self, -1, "" if settings["XAxisMax"]==None else str(settings["XAxisMax"]), style = wx.TE_LEFT , size =(150,-1))
        
        self.plotTitle = wx.TextCtrl(self, -1, settings["Title"], style = wx.TE_LEFT , size =(150,-1))
        self.plotXLabel = wx.TextCtrl(self, -1, settings["XLabel"], style = wx.TE_LEFT , size =(150,-1))
        self.plotYLabel = wx.TextCtrl(self, -1, settings["YLabel"], style = wx.TE_LEFT , size =(150,-1))
        self.plotXScale = wx.ComboBox(self, -1, size=(150, -1), choices=["Linear","Log"], style=wx.CB_READONLY)
        self.plotYScale = wx.ComboBox(self, -1, size=(150, -1), choices=["Linear","Log"], style=wx.CB_READONLY)
        self.plotXScale.SetSelection(0 if settings["XScale"]=="Linear" else 1)
        self.plotYScale.SetSelection(0 if settings["YScale"]=="Linear" else 1)
        
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        bagSizer = wx.GridBagSizer(10, 10)

        bagSizer.Add(plotTitleStatic,(0,0))
        bagSizer.Add(self.plotTitle, (0,1))
        
        bagSizer.Add(plotXAxisStatic,(1,0),(1,1))
        bagSizer.Add(plotXMinStatic,(2,0))
        bagSizer.Add(self.plotXAxisMin,(2,1))
        bagSizer.Add(plotXMaxStatic,(3,0))
        bagSizer.Add(self.plotXAxisMax,(3,1))
        bagSizer.Add(plotXLabelStatic,(4,0))
        bagSizer.Add(self.plotXLabel,(4,1))
        bagSizer.Add(plotXScaleStatic,(5,0))
        bagSizer.Add(self.plotXScale,(5,1))
        
        bagSizer.Add(plotYAxisStatic,(6,0),(1,1))
        bagSizer.Add(plotYMinStatic,(7,0))
        bagSizer.Add(self.plotYAxisMin,(7,1))
        bagSizer.Add(plotYMaxStatic,(8,0))
        bagSizer.Add(self.plotYAxisMax,(8,1))
        bagSizer.Add(plotYLabelStatic,(9,0))
        bagSizer.Add(self.plotYLabel,(9,1))
        bagSizer.Add(plotYScaleStatic,(10,0))
        bagSizer.Add(self.plotYScale,(10,1))
        
        
        
        bagSizer.AddGrowableCol(1)
        
        #Create OK and Cancel buttons
        buttonSizer =  self.CreateButtonSizer(wx.CANCEL|wx.OK)
        
        #Add information to the sizers
        mainSizer.Add(bagSizer,0,wx.ALL|wx.EXPAND,20)
        mainSizer.Add(buttonSizer,1,wx.ALL|wx.EXPAND,10)
 
        #Set the main sizer to the panel
        self.SetSizer(mainSizer)

        #Set size        
        mainSizer.Fit(self)
    
    def GetValues(self):
        
        xmax = self.plotXAxisMax.GetValue()
        xmin = self.plotXAxisMin.GetValue()
        ymax = self.plotYAxisMax.GetValue()
        ymin = self.plotYAxisMin.GetValue()
        
        title = self.plotTitle.GetValue()
        xlabel = self.plotXLabel.GetValue()
        ylabel = self.plotYLabel.GetValue()
        
        xscale = self.plotXScale.GetValue()
        yscale = self.plotYScale.GetValue()
        
        return xmax,xmin,ymax,ymin,title, xlabel, ylabel, xscale, yscale
        
class FilterPanel(wx.Panel):
    def __init__(self, parent,tree, **kwargs):
        wx.Panel.__init__( self, parent, **kwargs )
        
        #Store the parent
        self.parent = parent
        self.tree = tree
        self.active_filter = False
        
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        topBox = wx.StaticBox(self, label = "Filter")
        topSizer = wx.StaticBoxSizer(topBox, wx.VERTICAL)
        
        
        flexGrid = wx.FlexGridSizer(2, 1, 0, 10)
        
        #Create the checkboxes
        self.checkBoxParametersConstants = wx.CheckBox(self, -1, " Parameters / Constants")#, size=(140, -1))
        self.checkBoxTimeVarying = wx.CheckBox(self, -1, " Time-Varying", size=(140, -1))
        self.searchBox = wx.SearchCtrl(self, -1, "Search", size=(190, -1), style=wx.TE_PROCESS_ENTER)
        self.searchBox.SetToolTipString("Filter the variables using a unix filename pattern matching \n" \
                                         '(eg. "*der*"). Can also be a list of filters separated by ";"\n' \
                                         "See http://docs.python.org/2/library/fnmatch.html.")
        
        #Check the checkboxes
        self.checkBoxParametersConstants.SetValue(True)
        self.checkBoxTimeVarying.SetValue(True)
        
        #Add the checkboxes to the flexgrid
        flexGrid.Add(self.checkBoxParametersConstants)
        flexGrid.Add(self.checkBoxTimeVarying)
        flexGrid.Add(self.searchBox)

        flexGrid.AddGrowableCol(0)
        
        #Add information to the sizers
        topSizer.Add(flexGrid,1,wx.ALL|wx.EXPAND,10)
        mainSizer.Add(topSizer,0,wx.EXPAND|wx.ALL,10)
        
        #Set the main sizer to the panel
        self.SetSizer(mainSizer)
        
        #Bind events
        self.Bind(wx.EVT_CHECKBOX, self.OnParametersConstants, self.checkBoxParametersConstants)
        self.Bind(wx.EVT_CHECKBOX, self.OnTimeVarying, self.checkBoxTimeVarying)
        self.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN, self.OnSearch, self.searchBox)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnSearch, self.searchBox)
    
    def GetFilter(self):
        if self.active_filter == True:
            filter = self.searchBox.GetValue().split(";")
            if filter[0] == "": #If the filter is empty, match all
                filter = ["*"]
            filter_list = convert_filter(filter)
        else:
            filter_list = None
        return filter_list
    
    def OnSearch(self, event):
        self.active_filter = True
        self.tree.HideNodes(showTimeVarying=self.checkBoxTimeVarying.GetValue(), showParametersConstants=self.checkBoxParametersConstants.GetValue(), filter=self.GetFilter())
        
        if self.searchBox.GetValue() == "":
            self.active_filter = False
        
    def OnParametersConstants(self, event):
        self.tree.HideNodes(showTimeVarying=self.checkBoxTimeVarying.GetValue(), showParametersConstants=self.checkBoxParametersConstants.GetValue(), filter=self.GetFilter())
        
    def OnTimeVarying(self, event):
        self.tree.HideNodes(showTimeVarying=self.checkBoxTimeVarying.GetValue(), showParametersConstants=self.checkBoxParametersConstants.GetValue(), filter=self.GetFilter())

class Lines_Settings:
    def __init__(self, name=None):
        self.width = rcParams["lines.linewidth"]
        self.style = rcParams["lines.linestyle"]
        self.marker = rcParams["lines.marker"]
        self.markersize = rcParams["lines.markersize"]
        self.color = None
        self.name = name

class PlotPanel(wx.Panel):
    def __init__(self, parent, grid=False,move=True,zoom=False, **kwargs):
        wx.Panel.__init__( self, parent, **kwargs )

        #Initialize matplotlib
        self.figure = Figure(facecolor = 'white')
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.subplot = self.figure.add_subplot( 111 )

        self.parent = parent
        
        self.settings = {}
        self.settings["Grid"] = grid
        self.settings["Zoom"] = zoom
        self.settings["Move"] = move
        self.SetDefaultSettings() #Set the default settings
        
        self.plotVariables = []

        self._resizeflag = False

        self._SetSize()
        self.DrawSettings()
        
        #Bind events
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        
        #Bind event for resizing (must bind to canvas)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)
        
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.canvas.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.canvas.Bind(wx.EVT_LEAVE_WINDOW, self.OnLeaveWindow)
        self.canvas.Bind(wx.EVT_ENTER_WINDOW, self.OnEnterWindow)
        self.canvas.Bind(wx.EVT_MOTION, self.OnMotion)
        self.canvas.Bind(wx.EVT_LEFT_DCLICK, self.OnPass)
        
        self._mouseLeftPressed = False
        self._mouseMoved = False
    
    def SetDefaultSettings(self):
        self.settings["Title"] = ""
        self.settings["XLabel"] = "Time [s]"
        self.settings["YLabel"] = ""
        self.settings["XAxisMax"] = None
        self.settings["XAxisMin"] = None
        self.settings["YAxisMax"] = None
        self.settings["YAxisMin"] = None
        self.settings["XScale"] = "Linear"
        self.settings["YScale"] = "Linear"
        self.settings["LegendPosition"] = 0 #"Best" position
    
    def AddPlotVariable(self, ID, item, data):
        lineSettings = Lines_Settings(data["name"])
        self.plotVariables.append([ID,item,data,lineSettings])
        
    def GetPlotVariables(self):
        return self.plotVariables
        
    def DeleteAllPlotVariables(self):
        self.plotVariables = []
    
    def DeletePlotVariable(self, local_id=None, global_id=None):
        
        if local_id != None:
            for i,var in enumerate(self.plotVariables):
                if var[2]["variable_id"] == local_id:
                    self.plotVariables.pop(i)
                    break
                    
        if global_id != None:
            j = 0
            while j < len(self.plotVariables):
                if self.plotVariables[j][2]["result_id"] == global_id:
                    self.plotVariables.pop(j)
                    j = j-1
                j = j+1
                
                if j==len(self.plotVariables):
                    break
    """
    def DeletePlotVariable(self, item=None, ID=None):
        
        if item != None:
            for i,var in enumerate(self.plotVariables):
                if var[1]==item:
                    self.plotVariables.pop(i)
                    break
                    
        if ID != None:
            j = 0
            while j < len(self.plotVariables):
                if self.plotVariables[j][0] == ID:
                    self.plotVariables.pop(j)
                else:
                    if ID < self.plotVariables[j][0]:
                        self.plotVariables[j][0] = self.plotVariables[j][0]-1 
                    j = j+1
                    
                if j==len(self.plotVariables):
                    break
    """
    def OnPass(self, event):
        pass
    
    def OnMotion(self, event):
        
        if self._mouseLeftPressed: #Is the mouse pressed?
            self._mouseMoved = True
            self._newPos = event.GetPosition()
            
            if self.settings["Move"]:
                self.DrawMove()
            if self.settings["Zoom"]:
                self.DrawRectZoom()
    
    def DrawZoom(self):
        try:
            y0 = self._figureMin[1][1]-self._lastZoomRect[1]
            x0 = self._lastZoomRect[0]-self._figureMin[0][0]
            w = self._lastZoomRect[2]
            h = self._lastZoomRect[3]
            fullW = self._figureMin[1][0]-self._figureMin[0][0]
            fullH = self._figureMin[1][1]-self._figureMin[0][1]
            
            if w < 0:
                x0 = x0 + w
            x0 = max(x0, 0.0)
            y0 = max(y0, 0.0)
            
            plotX0 = self.subplot.get_xlim()[0]
            plotY0 = self.subplot.get_ylim()[0]
            plotW = self.subplot.get_xlim()[1]-self.subplot.get_xlim()[0]
            plotH = self.subplot.get_ylim()[1]-self.subplot.get_ylim()[0]
            
            self.settings["XAxisMin"] = plotX0+abs(x0/fullW*plotW)
            self.settings["XAxisMax"] = plotX0+abs(x0/fullW*plotW)+abs(w/fullW*plotW)
            self.settings["YAxisMin"] = plotY0+abs(y0/fullH*plotH)
            self.settings["YAxisMax"] = plotY0+abs(y0/fullH*plotH)+abs(h/fullH*plotH)
            
            self.DrawRectZoom(drawNew=False) #Delete the last zoom rectangle
            self.DrawSettings()
        except AttributeError:
            self.DrawRectZoom(drawNew=False) #Delete the last zoom rectangle
    
    def DrawMove(self):
        
        x0,y0 = self._originalPos
        x1,y1 = self._newPos
        
        fullW = self._figureMin[1][0]-self._figureMin[0][0]
        fullH = self._figureMin[1][1]-self._figureMin[0][1]
        
        plotX0,plotY0,plotW,plotH = self._plotInfo

        self.settings["XAxisMin"] = plotX0+(x0-x1)/fullW*plotW
        self.settings["XAxisMax"] = plotX0+plotW+(x0-x1)/fullW*plotW
        self.settings["YAxisMin"] = plotY0+(y1-y0)/fullH*plotH
        self.settings["YAxisMax"] = plotY0+plotH+(y1-y0)/fullH*plotH
        
        self.DrawSettings()    
    
    def DrawRectZoom(self, drawNew=True):
        dc = wx.ClientDC(self.canvas)
        dc.SetLogicalFunction(wx.XOR)

        wbrush =wx.Brush(wx.Colour(255,255,255), wx.TRANSPARENT)
        wpen =wx.Pen(wx.Colour(200, 200, 200), 1, wx.SOLID)
        dc.SetBrush(wbrush)
        dc.SetPen(wpen)


        dc.ResetBoundingBox()
        dc.BeginDrawing()
            
        y1 = min(max(self._newPos[1],self._figureMin[0][1]),self._figureMin[1][1])
        y0 = min(max(self._originalPos[1],self._figureMin[0][1]),self._figureMin[1][1])
        x1 = min(max(self._newPos[0],self._figureMin[0][0]),self._figureMin[1][0])
        x0 = min(max(self._originalPos[0],self._figureMin[0][0]),self._figureMin[1][0])

        if y1 > y0: 
            y0, y1 = y1, y0
        if x1 < y0: 
            x0, x1 = x1, x0

        w = x1 - x0
        h = y1 - y0

        rectZoom = int(x0), int(y0), int(w), int(h)

        try: 
            self._lastZoomRect
        except AttributeError: 
            pass
        else: 
            dc.DrawRectangle(*self._lastZoomRect)  #Erase last
        
        if drawNew:
            self._lastZoomRect = rectZoom
            dc.DrawRectangle(*rectZoom)
        else:
            try:
                del self._lastZoomRect
            except AttributeError:
                pass
        dc.EndDrawing()
        #dc.Destroy()
        
    def OnLeftDown(self, event):
        self._mouseLeftPressed = True #Mouse is pressed
        
        #Capture mouse position
        self._originalPos = event.GetPosition()
        #Capture figure size
        self._figureRatio = self.subplot.get_position().get_points()
        self._figureSize = (self.canvas.figure.bbox.width,self.canvas.figure.bbox.height)
        self._figureMin = [(round(self._figureSize[0]*self._figureRatio[0][0]),round(self._figureSize[1]*self._figureRatio[0][1])),
                           (round(self._figureSize[0]*self._figureRatio[1][0]),round(self._figureSize[1]*self._figureRatio[1][1]))]
        #Capture current plot
        plotX0 = self.subplot.get_xlim()[0]
        plotY0 = self.subplot.get_ylim()[0]
        plotW = self.subplot.get_xlim()[1]-self.subplot.get_xlim()[0]
        plotH = self.subplot.get_ylim()[1]-self.subplot.get_ylim()[0]
        self._plotInfo = (plotX0, plotY0, plotW, plotH)
        
        
    def OnLeftUp(self, event):
        self._mouseLeftPressed = False #Mouse is not pressed
        if self._mouseMoved:
            self._mouseMoved = False
            
            if self.settings["Zoom"]:
                self.DrawZoom()
            if self.settings["Move"]:
                self.DrawMove()

    def OnLeaveWindow(self, event): #Change cursor
        if self._mouseLeftPressed:
            self._mouseLeftPressed = False #Mouse not pressed anymore
            self._mouseMoved = False
            
            if self.settings["Zoom"]:
                self.DrawZoom()
            if self.settings["Move"]:
                self.DrawMove()
        
    def OnEnterWindow(self, event): #Change cursor
        self.UpdateCursor()
    
    def OnRightDown(self, event):
        """
        On right click, resize the plot.
        """
        self.ReSize()
        
    def ReSize(self):
        self.UpdateSettings(axes=[None,None,None,None])
        self.DrawSettings()

    def OnSize(self, event):
        self._resizeflag = True

    def OnIdle(self, event):
        if self._resizeflag:
            self._resizeflag = False
            self._SetSize()

    def _SetSize(self):
        pixels = tuple(self.GetClientSize())
        #self.SetSize(pixels) #GENERATES INFINITELY EVENTS ON UBUNTU
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0])/self.figure.get_dpi(),
                                    float(pixels[1])/self.figure.get_dpi())
                                     
    #def Draw(self, variables=[]):
    def Draw(self):
        self.subplot.clear()
        self.subplot.hold(True)

        for i in self.plotVariables:
            traj = i[2]["traj"].get_variable_data(i[2]["full_name"])
            if i[3].color is None:
                #self.subplot.plot(i[2]["traj"].t, i[2]["traj"].x,label=i[3].name,linewidth=i[3].width,marker=i[3].marker,linestyle=i[3].style,markersize=i[3].markersize)
                self.subplot.plot(traj.t, traj.x,label=i[3].name,linewidth=i[3].width,marker=i[3].marker,linestyle=i[3].style,markersize=i[3].markersize)
            else:
                #self.subplot.plot(i[2]["traj"].t, i[2]["traj"].x,label=i[3].name,linewidth=i[3].width,marker=i[3].marker,linestyle=i[3].style,markersize=i[3].markersize,color=i[3].color)
                self.subplot.plot(traj.t, traj.x,label=i[3].name,linewidth=i[3].width,marker=i[3].marker,linestyle=i[3].style,markersize=i[3].markersize,color=i[3].color)
                
        self.DrawSettings()
        
    def GetLines(self):
        return self.subplot.get_lines()
        
    def GetLegendLocation(self):
        res = self.subplot.get_legend()
        
        if res is None:
            return -1
        else:
            return res._loc

    def Save(self, filename):
        """
        Saves the current figure.
        
        Parameters::
        
            filename - The name of the to be saved plot.
        """
        self.figure.savefig(filename)
        
    def DrawSettings(self):
        """
        Draws the current settings onto the Plot.
        """
        
        self.subplot.grid(self.settings["Grid"])

        #Draw label settings
        self.subplot.set_title(self.settings["Title"])
        self.subplot.set_xlabel(self.settings["XLabel"])
        self.subplot.set_ylabel(self.settings["YLabel"])
        
        #Draw Scale settings
        self.subplot.set_xscale(self.settings["XScale"])
        self.subplot.set_yscale(self.settings["YScale"])
        
        if len(self.plotVariables) != 0 and self.settings["LegendPosition"] != -1:
            self.subplot.legend(loc=self.settings["LegendPosition"])
        
        #Draw axis settings
        if self.settings["XAxisMin"] != None:
            #self.subplot.set_xlim(left=self.settings["XAxisMin"])
            self.subplot.set_xlim(xmin=self.settings["XAxisMin"])
        if self.settings["XAxisMax"] != None:
            #self.subplot.set_xlim(right=self.settings["XAxisMax"])
            self.subplot.set_xlim(xmax=self.settings["XAxisMax"])
        if self.settings["XAxisMax"] == None and self.settings["XAxisMin"] == None:
            self.subplot.set_xlim(None,None)
            self.subplot.set_autoscalex_on(True)
            #self.subplot.autoscale(axis="x")
            self.subplot.autoscale_view(scalex=True)
        
        if self.settings["YAxisMin"] != None:
            #self.subplot.set_ylim(bottom=self.settings["YAxisMin"])
            self.subplot.set_ylim(ymin=self.settings["YAxisMin"])
        if self.settings["YAxisMax"] != None:
            #self.subplot.set_ylim(top=self.settings["YAxisMax"])
            self.subplot.set_ylim(ymax=self.settings["YAxisMax"])
        if self.settings["YAxisMax"] == None and self.settings["YAxisMin"] == None:
            self.subplot.set_ylim(None,None)
            self.subplot.set_autoscaley_on(True)
            #self.subplot.autoscale(axis="y") #METHOD DOES NOT EXIST ON VERSION LESS THAN 1.0
            self.subplot.autoscale_view(scaley=True)

        #Draw
        self.canvas.draw()
        
    def UpdateSettings(self, grid=None, title=None, xlabel=None,
                        ylabel=None, axes=None, move=None, zoom=None,
                        xscale=None, yscale=None, legendposition=None):
        """
        Updates the settings dict.
        """
        if grid !=None:
            self.settings["Grid"] = grid
        if title !=None:
            self.settings["Title"] = title
        if xlabel !=None:
            self.settings["XLabel"] = xlabel
        if ylabel !=None:
            self.settings["YLabel"] = ylabel
        if axes != None:
            self.settings["XAxisMin"]=axes[0]
            self.settings["XAxisMax"]=axes[1]
            self.settings["YAxisMin"]=axes[2]
            self.settings["YAxisMax"]=axes[3]
        if move != None:
            self.settings["Move"] = move
        if zoom != None:
            self.settings["Zoom"] = zoom
        if xscale != None:
            self.settings["XScale"] = xscale
        if yscale != None:
            self.settings["YScale"] = yscale
        if legendposition != None:
            self.settings["LegendPosition"] = legendposition

    def UpdateCursor(self):
        if self.settings["Move"]:
            cursor = wx.StockCursor(wx.CURSOR_HAND)
            self.canvas.SetCursor(cursor)
        if self.settings["Zoom"]:
            cursor = wx.StockCursor(wx.CURSOR_CROSS)
            self.canvas.SetCursor(cursor)
        
    def GetSettings(self):
        """
        Returns the settigns of the current plot.
        """
        return self.settings
    

def startGUI(filename=None):
    """
    Starts GUI.
    If a filename is provided, that file is loaded into the GUI on startup.
    """
    #Start GUI
    app = wx.App(False)
    gui = MainGUI(None, -1,filename)
    app.MainLoop()

if __name__ == '__main__':
    startGUI()
