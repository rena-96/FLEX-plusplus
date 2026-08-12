
_initialized = False


def initialize_module():
    global _initialized
    if not _initialized:
        print("""
             ██████  ██      ██████  ██   ██    ██      ██  
             ██      ██      ██       ██ ██     ██      ██  
             █████   ██      ████      ███    ██████  ██████
             ██      ██      ██       ██ ██     ██      ██  
             ██      ██████  ██████  ██   ██    ██      ██
             
             AB INITIO MD · SLOW GROWTH · APPLIED POTENTIAL       """)
        print('Fluctuation-Dissipation Extension for VASPsol++')

        print('      Copyright © 2026 Technical University of Denmark      ')
        print '      This version of the software was developed by \n Renata Sechi, Ph.D., and Ronan Gleeson, Msc.'
        print('      Please cite: Chemrxiv  ')

        _initialized = True


initialize_module()

