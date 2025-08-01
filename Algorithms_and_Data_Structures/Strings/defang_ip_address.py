'''
a defanged IP address has its every period '.' replaced
with '[.]'

given a valid IPv4 address, we return its defanged version
'''

class defang:
    def defang_ip_address(self, s: str) -> str:
        s = s.replace(".", "[.]")
        return s

#example usage
if __name__=='__main__':
    
    #create an instance of the defang class
    defanger = defang()
    
    #call the defang_ip_address method
    result = defanger.defang_ip_address("127.0.0.0")
    print(result)