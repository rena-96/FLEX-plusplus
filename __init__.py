# Initialize a flag to track if the message has been printed
_initialized = False

# Define a function to print the message once
def initialize_module():
    global _initialized
    if not _initialized:
        print("""
███████        ██████ ██               ██████  
 ██            ██      ██              ██    ██ 
 █████         ██      ███████         ██    ██ 
 ██            ██      ██   ██         ██    ██ 
 ███████ lectro ██████ ██   ██ emical   ██████ ptimizer
        """)

        print('             Copyright © 2026 Renata Sechi, Ronan Gleeson')
        print('       Please cite: Chemrxiv  ')

        _initialized = True

# Call the function during the first import
initialize_module()

# The rest of your module initialization can go here if needed
