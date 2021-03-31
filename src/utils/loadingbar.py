def print_progress(current_progress, total_progress, description='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current_progress / float(total_progress)))
    filledLength = int(length * current_progress // total_progress)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{description} |{bar}| {percent}% {suffix}', end="\t")

    if current_progress == total_progress:
        print()
