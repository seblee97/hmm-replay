#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>

#include <map>
#include <any>
#include <stdexcept>

// using namespace std;

template <typename Out>
void split(const std::string &s, char delim, Out result)
{
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim))
    {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::map<std::string, std::any>
parse_input(std::string input_file_path)
{
    std::map<std::string, std::any> config;

    std::ifstream ifs; // input file stream
    std::string str;
    ifs.open(input_file_path,
             std::ios::in); // input file stream

    if (ifs)
    {
        while (!ifs.fail())
        {
            std::getline(ifs, str);

            const std::string string_line = str;
            std::vector<std::string>
                split_string = split(string_line, ';');

            std::any value;

            if (split_string[0] == "int")
            {
                int value = stoi(split_string[2]);
            }
            else if (split_string[0] == "float")
            {
                float value = std::stof(split_string[2]);
            }
            else if (split_string[0] == "str")
            {
                std::string value = split_string[2];
            }
            else
            {

                throw std::invalid_argument("type not recognised.\n");
            }

            config[split_string[1]] = value;
            {
                std::cout << "type " << split_string[0] << " key " << split_string[1] << " value " << split_string[2] << std::endl;
            }
        }
        ifs.close();
    }

    return config;
}

int main(int argc, char **argv)
{
    std::cout << "Have " << argc << " arguments:" << std::endl;

    for (int i = 0; i < argc; ++i)
    {
        std::cout << argv[i] << std::endl;
    }

    std::string input_file_path = argv[1];
    std::string output_file_path = argv[2];

    std::map<std::string, std::any> config = parse_input(input_file_path);
    std::vector<std::string> key;
    std::vector<std::any> value;

    for (auto const &imap : config)
    {
        key.push_back(imap.first);
        value.push_back(imap.second);
        std::cout << "Key: " << imap.first << std::endl;
    }
}