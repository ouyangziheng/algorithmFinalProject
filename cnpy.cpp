#include "cnpy.h"

namespace cnpy {

char BigEndianTest() {
    int x = 1;
    return (((char*)&x)[0]) ? '<' : '>';
}

char map_type(const std::type_info& t) {
    if (t == typeid(float)) return 'f';
    if (t == typeid(double)) return 'f';
    if (t == typeid(long double)) return 'f';

    if (t == typeid(int)) return 'i';
    if (t == typeid(char)) return 'i';
    if (t == typeid(short)) return 'i';
    if (t == typeid(long)) return 'i';
    if (t == typeid(long long)) return 'i';

    if (t == typeid(unsigned char)) return 'u';
    if (t == typeid(unsigned short)) return 'u';
    if (t == typeid(unsigned long)) return 'u';
    if (t == typeid(unsigned long long)) return 'u';
    if (t == typeid(unsigned int)) return 'u';

    if (t == typeid(bool)) return 'b';

    if (t == typeid(std::complex<float>)) return 'c';
    if (t == typeid(std::complex<double>)) return 'c';
    if (t == typeid(std::complex<long double>))
        return 'c';

    else
        return '?';
}

// 辅助函数：将字符串添加到vector<char>
void append_to_vector(std::vector<char>& vec, const std::string& str) {
    vec.insert(vec.end(), str.begin(), str.end());
}

// 辅助函数：将C风格字符串添加到vector<char>
void append_to_vector(std::vector<char>& vec, const char* str) {
    vec.insert(vec.end(), str, str + std::strlen(str));
}

// 辅助函数：将单个字符添加到vector<char>
void append_to_vector(std::vector<char>& vec, char c) { vec.push_back(c); }

template <typename T>
std::vector<char> create_npy_header(const std::vector<size_t>& shape) {
    std::vector<char> dict;
    append_to_vector(dict, "{'descr': '");
    append_to_vector(dict, BigEndianTest());
    append_to_vector(dict, map_type(typeid(T)));
    append_to_vector(dict, std::to_string(sizeof(T)));
    append_to_vector(dict, "', 'fortran_order': False, 'shape': (");
    append_to_vector(dict, std::to_string(shape[0]));
    for (size_t i = 1; i < shape.size(); i++) {
        append_to_vector(dict, ", ");
        append_to_vector(dict, std::to_string(shape[i]));
    }
    if (shape.size() == 1) append_to_vector(dict, ",");
    append_to_vector(dict, "), }");
    // pad with spaces so that preamble+dict is modulo 64 bytes. preamble is 10
    // bytes. dict needs to end with \n
    int remainder = 64 - (10 + dict.size() + 1) % 64;
    dict.insert(dict.end(), remainder, ' ');
    dict.push_back('\n');

    // create the npy header
    std::vector<char> header;
    append_to_vector(header, "\x93NUMPY");
    header.push_back((char)0x01);  // major version
    header.push_back((char)0x00);  // minor version
    unsigned short header_len = dict.size();
    header.push_back((char)(header_len & 0xFF));
    header.push_back((char)((header_len >> 8) & 0xFF));
    header.insert(header.end(), dict.begin(), dict.end());

    return header;
}

void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape,
                      bool& fortran_order) {
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    if (header[header.size() - 1] != '\n')
        throw std::runtime_error(
            "parse_npy_header: header missing terminating newline");

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error(
            "parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error(
            "parse_npy_header: failed to find header keyword: '(' or ')'");

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    if (str_shape[str_shape.size() - 1] == ',')
        str_shape = str_shape.substr(0, str_shape.size() - 1);

    size_t ndims = 1;
    size_t pos = 0;
    shape.clear();
    for (size_t i = 0; i < str_shape.size(); i++) {
        if (str_shape[i] == ',') ndims++;
    }
    for (size_t i = 0; i < ndims; i++) {
        loc1 = str_shape.find(",", pos);
        if (loc1 == std::string::npos) loc1 = str_shape.size();
        shape.push_back(std::stoul(str_shape.substr(pos, loc1 - pos)));
        pos = loc1 + 1;
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error(
            "parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian =
        (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);  // TODO: Handle big endian

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = std::stoul(str_ws.substr(0, loc2));
}

void parse_npy_header(unsigned char* buffer, size_t& word_size,
                      std::vector<size_t>& shape, bool& fortran_order) {
    // std::string magic_string(buffer,6);
    // uint8_t major_version = buffer[6];
    // uint8_t minor_version = buffer[7];
    unsigned short header_len = *reinterpret_cast<unsigned short*>(buffer + 8);
    std::string header(reinterpret_cast<char*>(buffer + 10), header_len);

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order") + 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    if (str_shape[str_shape.size() - 1] == ',')
        str_shape = str_shape.substr(0, str_shape.size() - 1);

    size_t ndims = 1;
    size_t pos = 0;
    shape.clear();
    for (size_t i = 0; i < str_shape.size(); i++) {
        if (str_shape[i] == ',') ndims++;
    }
    for (size_t i = 0; i < ndims; i++) {
        loc1 = str_shape.find(",", pos);
        if (loc1 == std::string::npos) loc1 = str_shape.size();
        shape.push_back(std::stoul(str_shape.substr(pos, loc1 - pos)));
        pos = loc1 + 1;
    }

    // endian, word size, data type
    loc1 = header.find("descr") + 9;
    bool littleEndian =
        (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);  // TODO: Handle big endian

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = std::stoul(str_ws.substr(0, loc2));
}

NpyArray npy_load(std::string fname) {
    FILE* fp = fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);

    NpyArray arr;
    size_t word_size;
    bool fortran_order;
    parse_npy_header(fp, word_size, arr.shape, fortran_order);
    arr.word_size = word_size;
    arr.fortran_order = fortran_order;
    arr.num_vals = 1;
    for (size_t i = 0; i < arr.shape.size(); i++) arr.num_vals *= arr.shape[i];

    arr.data_holder = std::shared_ptr<std::vector<char>>(
        new std::vector<char>(arr.num_vals * arr.word_size));

    size_t nread =
        fread(&(*arr.data_holder)[0], arr.word_size, arr.num_vals, fp);
    if (nread != arr.num_vals)
        throw std::runtime_error("npy_load: failed fread");

    fclose(fp);
    return arr;
}

}  // namespace cnpy