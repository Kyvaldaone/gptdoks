# gptdoks
итоги того как вайбкодинг влияет на психику:

# Краткое описание
Сочетание скрипта+промта для чатгпт, общая цель которых - собирать
и характеризовать информацию о людях на основах json скачанных с телеграмма

# Документация
## 1.0 Telegram
Для начала вам нужно зайти в Telegram Dekstop
(именно декстопная версия, мобильная не подойдет), в большинстве чатов
при нажатии трех точек справа теперь будет новая вкладка export chat history,
вам нужно нажать на эту вкладку и в параметрах желательно сразу поставить size limit
 на 4000мб чтобы чат точно скачался полностью, также нужно убрать все галочки в медиа,
 то есть фото и все прочее, также будет подсвеченный текст возле format: html, вам нужно нажать на html
 и вместо html выбрать json и нажать кнопку начала скачивания, настройки времени желательно не менять
 так как они все равно не работают, после скачивания у вас появится кнопка сразу
 перейти в директорию где будет лежать ваш json

## 1.1 Script by Kyvalda (вайбкод)
Используйте файл xapakter.py прикрепленный к репозиторию,
перед началом работы с файлом убедитесь что у вас есть текстовый редактор
на примере notepad++, а также пропишите в cmd, pip install tqdm tiktoken (одна команда),
после установки текстового редактора и библиотек откройте вышеуказанный файл
через текстовый редактор и обратите внимание на строки 12 и 14, а конкретно
это TARGET_NAME = "nickname" и OUTPUT_FILE = ".txt", в первом вы должны в кавычках
(кавычки не убирать) указать никнейм (НЕ ТЕГ А ИМЕННО ОТОБРАЖАЕМЫЙ НИК) человека сообщения
которого вы будете собирать, рекомендуется проверять ник через чат чтоб не ошибиться, особые
символы тоже должны быть указаны в скрипте если они есть в нике, а в output_file укажите
название для файла который вы в итоге получите, рекомендуется делать разные для разных ников,
когда вы все это сделали, сохраните файл через текстовый редактор и через консоль открытую в этой
папке (пкм по пустому месту и "открыть в терминале") пропишите python xapakter.py, программа начнет
свою работу что может занять от пары секунд до 5 минут, по итогу вы получите уведомление о создании файла
и информацию о числе нейросетевых токенов, обратите внимание на это число, если это число будет более 1000000 (миллиона)
то даже не пытайтесь приступать к следующему этапу, вам прийдется сокращать либо вручную либо другими
методами полученный файл, однако если токенов менее миллиона то можете приступать к следующему ходу

## 1.2 ChatGPT и промт (ну хоть промт сделан вручную)
Откройте ChatGPT (если не доступен в вашей стране используйте VPN) 
(важно использовать именно чатгпт так как другие нейросети 
не имеют столь большего контекстного окна и не смогут нормально
просмотреть весь текст вашего файла) - создайте новый чат
(ОБЯЗАТЕЛЬНО, в будущем когда будете делать много сборов инфы
для каждого делайте отдельный чат чтоб не перемешивать инфу, 
также учитывайте лимит чатгпт в 3-10 файлов после которых ставится лимит
на 24 часа и вам прийдется заходить с другого аккаунта), и в строку отправки
напишите следующий промт с 1.21

после того как написали промт - перетяните в вкладку чата
файл который вы получили после работы скрипта, и только после этого делайте отправку запроса

## 1.21 Промт

привет, у меня есть огромнейший текстовый файл со всеми сообщениями человека с ником
(укажите ник без скобок), я почти ничего о этом человеке не знаю
кроме того что это (укажите пол без скобок, парень либо девушка), 
можешь разобрать характер на основе всех сообщений и выдать потверждения или новые анализы характера по следующим показателям

 возвраст в годах
 
 страна
 
 город
 
 Языки которые понимает и на которых может разговаривать
 
 Потенциальная работа
 
 Возможная специальность на которую учился или учится
 
 семейное положение
 
 любимые игры и жанры
 
 любимый цвет (если найдешь) :
 
 любимое животное:
 
 Возможно фетиши или подобное связанное с нравами в нсфв если возможно: (ПИШИ ЭТОТ ПУНКТ ОЧ ПОДРОБНО)
 
 отношение к мужчинам:
 
 отношение к женщинам:
 
 отношение к гетеро:
 
 отношение к другим ориентациям:
 
 отношение к трансам:
 
 ориентация на основе сообщений:
 
 Поддержка власти своей страны или отсутствие поддержки:
 
 Политическое мнение о ближайших и далеких странах:
 
 Общая "сфера влияния" под которой мысли (например про рос про укр и прочее):
 
 Политическая идеология:
 
 Любимые языки программирования (если занимается этим):
 
Также сделай внизу отчета доп отчет где по каждому сообщению будет
написано одно из сообщений пруфов на основе которого ты решал определенные пункты
в общем анализ характера, чекай весь текст и все сообщения
сделай два раздела, сначала подробный текстовый анализ по каждому пункту
потом маленькая табличка с общей инфой и пруф сообщениями

## 1.3 Готово)
Вы получите довольно большой отчет с пруф сообщениями на большинство тезисов, 
также можете дополнять промт своими пунктами или убирать ненужные.
