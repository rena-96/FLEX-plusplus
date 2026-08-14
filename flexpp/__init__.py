_initialized = False


def initialize_module():
    """Print the FLEX++ banner, copyright and citation notice once per session.

    Called at import time; the module-level `_initialized` flag makes repeated
    calls no-ops so re-importing the package does not print the banner again.
    """
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
        print('      This version of the software was developed by')
        print('      Renata Sechi, Ph.D., and Ronan Gleeson, Msc.')
        print('      Please cite: Chemrxiv  ')

        _initialized = True


initialize_module()
