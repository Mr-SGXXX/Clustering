# Copyright (c) 2023 Yuxuan Shao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import configparser
import typing


class CaseSensitiveConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def optionxform(self, optionstr):
        return optionstr

class config:
    """
    Configuration Class
    
    Attributes:
        cfg: configparser.ConfigParser, configuration object
        split_symbol: str, delimiter of the configuration file
        
        Methods:
            init_by_path: initialize the configuration object by the configuration file path
            get: get the value of the option in the configuration file
            __str__: print the configuration file        
            
    """
    def __init__(self, cfg:CaseSensitiveConfigParser, split_symbol=','):
        self.cfg = cfg
        self.split_symbol = split_symbol
    
    @classmethod
    def init_by_path(self, config_path, split_symbol=','):
        """
        Initialize the configuration object by the configuration file path.
        :param config_path: str, configuration file path
        :param split_symbol: str, delimiter of the configuration file
        """
        cfg = CaseSensitiveConfigParser()
        cfg.read(config_path)
        return self(cfg, split_symbol)
    
    def set(self, session:str=None, option:str=None, value:typing.Any=None):
        """
        Set the value of the option in the configuration file.
        :param session: str, session name
        :param option: str, option name
        :param value: typing.Any, value of the option
        """
        self.cfg.set(session, option, value)

    def get(self, session:str=None, option:str=None):
        """
        Get the value of the option in the configuration file. It will automatically convert the string to the corresponding type.
        :param session: str, session name
        :param option: str, option name
        """
        def is_float(str_data:str):
            if '.' in str_data:
                parts = str_data.split('.')
                if len(parts) == 2 and (parts[0].isdigit() or parts[0] == '') and (parts[1].isdigit() or parts[1] == ''):
                    return True
            elif 'e' in str_data.lower():
                parts = str_data.replace('-', "").lower().split('e')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return True
            return False
        
        def is_int(str_data):
            if str_data.isdigit():
                return True
            return False
        
        def is_None(str_data):
            if str_data == "None":
                return True
            return False
        
        def is_Bool(str_data):
            if str_data == "True" or str_data == "False":
                return True
            return False
        
        def process(str_data:str):
            if self.split_symbol not in str_data:
                if is_None(str_data):
                    return None
                elif is_float(str_data):
                    return float(str_data)
                elif is_int(str_data):
                    return int(str_data)
                elif is_Bool(str_data):
                    return True if str_data == "True" else False
                else:
                    return str_data
            else:
                return [process(part.strip()) for part in str_data.split(self.split_symbol)]
        
        str_data = self.cfg.get(session, option)
        return process(str_data)
    
    def __getitem__(self, section_name)->dict:
        return {option: self.get(section_name, option) for option in self.cfg.options(section_name)}

    def __len__(self):
        return len(self.cfg.sections())
    
    def __str__(self) -> str:
        rst_str = "Config Setting:\n\n"
        for section in self.cfg.sections():
            if section == "global" or section == self.get("global", "method_name") or section == self.get("global", "dataset"):
                rst_str += f'[{section}]\n'
                for option in self.cfg.options(section):
                    if "pwd" not in option and "password" not in option:
                        value = self.cfg.get(section, option)
                        rst_str += f'{option}:\t{value}\n'
                rst_str += "\n"
        return rst_str