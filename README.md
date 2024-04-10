# Детектор лиц, глаз и улыбок с использованием каскадов Хаара в OpenCV

Этот проект представляет собой программу на C++, которая позволяет обнаруживать лица, глаза и улыбки на видео с помощью каскадов Хаара и библиотеки OpenCV. Программа использует параллельную обработку с помощью OpenMP для ускорения обнаружения объектов.

## Требования
- C++ компилятор поддерживающий стандарт C++11
- Библиотека OpenCV

## Как использовать
1. Установите необходимые зависимости (C++ компилятор, OpenCV). 
2. Скомпилируйте программу с использованием компилятора, поддерживающего OpenMP 
3. Запустите программу и укажите путь к видео для обработки 
4. Наслаждайтесь результатом обнаружения лиц, глаз и улыбок на видео 

## Дополнительные информация
Параметры обнаружения объектов (лиц, глаз, улыбок) и пути к каскадам Хаара указаны в коде программы.

Файлы каскадов Хаара должны быть доступны по указанным путям для успешной работы программы.
