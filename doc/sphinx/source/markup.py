
def mark_examples():
    import pyfmi.examples as examples
    
    exclude = ["log_analysis"]
    
    for ex in examples.__all__:
        
        if ex in exclude:
            continue
            
        file = open("EXAMPLE_"+ex+".rst",'w')
        
        file.write(ex + '.py\n')
        file.write('===================================\n\n')
        file.write('.. autofunction:: pyfmi.examples.'+ex+'.run_demo\n')
        file.write('   :noindex:\n\n')
        file.write('.. note::\n\n')
        file.write('    Press [source] (to the right) to view the example.\n')
        file.close()


mark_examples()
