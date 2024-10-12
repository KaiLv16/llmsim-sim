

def save_first_xxx_lines(lines, input_file):
    output_file = input_file + '_' + str(lines) + '.txt'
    input_file = input_file + '.txt'
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            # 读取前10000行
            lines = [next(infile) for _ in range(lines)]
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # 将读取到的行写入新文件
            outfile.writelines(lines)
        
        print(f"Successfully saved the first 10000 lines to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except StopIteration:
        print(f"Warning: {input_file} has less than 10000 lines. Saved available lines to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    save_first_xxx_lines(1000000, 'mix/output/test_0/test_0_snd_rcv_record_file')

