from langchain_core.tools import tool

@tool
def addNumbers(a:int,b:int)->str:
    '''addition of 2 numbers'''
    return f'Sum of 2 Number is ={(a+b)}'

@tool
def multiplyNumbers(a:int,b:int)->str:
    '''multipication of 2 numbers'''
    return f'Multiply Result={(a*b)}'

@tool
def saveFile(content:str)->str:
    '''saving content to local file i.e hello.txt'''
    with open("./hello.txt","w+") as file:
        file.write(content)
        file.close()
        return 'File created successfully'

@tool
def readFile()->str:
    '''reading from created file i.e hello.txt'''
    with open("./hello.txt","r+") as file:
        data = file.read()
        file.close()
        return data
