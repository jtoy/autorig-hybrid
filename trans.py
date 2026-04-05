from processing import change_alpha

def main():
    black_bg = 'diecut_black.png'
    white_bg = 'diecut_white.png'
    output = 'diecut_transparent.png'
    change_alpha(black_bg, white_bg, output, threshold=30)

if __name__ == "__main__":
    main()