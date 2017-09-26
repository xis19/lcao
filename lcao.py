#! /usr/bin/env python3

"""
LeetCode at Office
"""

import ast
import logging
import os.path
import sys
import time
import typing

from collections import namedtuple
from urllib.parse import urljoin

import bs4
import click
import dogpile.cache
import requests
import requests.sessions
import requests.cookies

from termcolor import cprint

FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('lcao')

USER_DIRECTORY = os.path.expanduser('~')
CONFIG_DIRECTORY = os.path.join(USER_DIRECTORY, '.config', 'lcao')

file_cache_region = dogpile.cache.make_region(name='lcao')
memory_cache_region = dogpile.cache.make_region(name='lcao.memory').configure(
    backend='dogpile.cache.memory',
    expiration_time=300
)


class _Endpoints(object):
    BASE_URL = 'https://leetcode.com'
    HOST = 'leetcode.com'
    LOGIN = urljoin(BASE_URL, 'accounts/login')
    LOGOUT = urljoin(BASE_URL, 'accounts/logout')
    ALL_PROBLEMS = urljoin(BASE_URL, 'api/problems/all')
    SUBMISSIONS = urljoin(BASE_URL, 'submissions/')

    @staticmethod
    def problem_description(problem_slug: str) -> str:
        return urljoin(_Endpoints.BASE_URL, 'problems/{}/description/'.format(problem_slug))

    @staticmethod
    def run_code(problem_slug: str) -> str:
        return urljoin(_Endpoints.BASE_URL, 'problems/{}/interpret_solution/'.format(problem_slug))

    @staticmethod
    def submit_code(problem_slug: str) -> str:
        return urljoin(_Endpoints.BASE_URL, 'problems/{}/submit/'.format(problem_slug))

    @staticmethod
    def check_run_code_status(interpret_id: str) -> str:
        return urljoin(
            _Endpoints.BASE_URL,
            'submissions/detail/{}/check'.format(interpret_id)
        )


_UserInformation = namedtuple('_UserInformationTuple', [
    'name', 'num_solved', 'ac_easy', 'ac_medium', 'ac_hard', 'subscribed'
])

_AllProblemsInformation = namedtuple('_ProblemInformation', [
    'total_num', 'num_easy', 'num_medium', 'num_hard'
])

_ProblemStatisticalInformation = namedtuple('_ProblemStatisticalInformation', [
    'total_submitted', 'total_accepts', 'frequency'
])

_ProblemListItem = namedtuple('_ProblemListItem', [
    'index', 'title', 'slug', 'difficulty', 'statistical_info', 'paid_only', 'status'
])

_ProblemInformation = namedtuple('_ProblemInformation', [
    'index', 'title', 'description', 'code_definition', 'sample_testcase'
])

_TestRunResultSuccess = namedtuple('_TestRunResultSuccess', [
    'test_case', 'expected', 'expected_time', 'actual', 'actual_time', 'code_output'
])

_TestRunResultFailure = namedtuple('_TestRunResultFailure', [
    'error', 'test_case', 'code_output'
])

_SubmitResultAccepted = namedtuple('_SubmitResultSuccess', [
    'time', 'stdout', 'total_test_cases'
])

_SubmitResultWrongAnswer = namedtuple('_SubmitResultWrongAnswer', [
    'total_test_cases', 'total_correct', 'stdout', 'last_failed_test_case', 'last_expected', 'last_actual',
    'test_results'
])

_SubmitResultFailure = namedtuple('_SubmitResultFailure', [
    'error', 'stdout', 'last_failed_test_case'
])

_RUN_ERROR_STATUS_CODE = {
    12: 'Memory Limit Exceeded',
    14: 'Time Limit Exceeded',
    15: 'Runtime Error',
    20: 'Compile Error'
}

_DIFFICULTY_NAME_MAPPER = {
    1: 'Easy',
    2: 'Medium',
    3: 'Hard'
}

_DIFFICULTY_COLOR_MAPPER = {
    1: 'green',
    2: 'magenta',
    3: 'red'
}

_STATUS_NAME_MAPPER = {
    None: '-',
    'notac': '?',
    'ac': '*'
}

_STATUS_COLOR_MAPPER = {
    None: 'grey',
    'notac': 'red',
    'ac': 'green'
}


class _PrintFormatter(object):
    STATUS = ' {:1s} '
    INDEX = '{:5s}'
    TITLE = '{:60s}'
    DIFFICULTY = '{:12s}'
    ACCEPTANCE_RATE = '{:12s}'
    SUBSCRIBE_REQUIRED = '{:10s}'


_LIST_PROBLEM_FORMATTER = ''.join([
    _PrintFormatter.STATUS,
    _PrintFormatter.INDEX,
    _PrintFormatter.TITLE,
    _PrintFormatter.DIFFICULTY,
    _PrintFormatter.ACCEPTANCE_RATE,
    _PrintFormatter.SUBSCRIBE_REQUIRED
])

_LANGUAGE_EXT_MAPPER = {
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'csharp',
    '.go': 'golang',
    '.java': 'java',
    '.js': 'javascript',
    '.py': 'python',
    '.rb': 'ruby',
    '.scala': 'scala',
    '.sql': 'mysql'
}


def _parse_raw_problem_html(html_data: str) -> _ProblemInformation:
    parser = bs4.BeautifulSoup(html_data, 'html.parser')

    detail_container = parser.select_one('div[class="container question-detail-container"]')

    caption = detail_container.select_one('h3')
    index, title = caption.text.strip().split('. ')

    description = detail_container.select_one('div[class="question-description"]').text.strip()

    # Code definition is hardcoded in the JavaScript part... How sad we have to use regexp to capture them
    # Brutal force search for the <script> element with the code definition
    code_definition = {}
    sample_test_case = ''

    scripts = parser.select('script')
    interested_script_element = None
    for script in scripts:
        if 'var pageData' in script.text:
            interested_script_element = script
            break

    if interested_script_element:
        for line in interested_script_element.text.split('\n'):
            if 'codeDefinition' in line:
                _, definition_raw = line.split(': ', 1)
                # Remove tailing ',' as it is part of a JSON object
                definition_raw = definition_raw.strip(',')
                code_definition = {
                    item['value']: item['defaultCode']
                    for item in ast.literal_eval(definition_raw)
                }
            if 'sampleTestCase' in line:
                _, sample_test_case = line.split(': ', 1)
                # Remove tailing ',', as it is part of a JSON object
                sample_test_case = ast.literal_eval(sample_test_case.strip(','))

    return _ProblemInformation(
        index=int(index),
        title=title,
        description=description,
        code_definition=code_definition,
        sample_testcase=sample_test_case
    )


class LeetCode(object):
    def __init__(self) -> None:
        self._session_ = None

    @property
    def _session(self) -> requests.Session:
        if not self._session_:
            persisted_cookies = file_cache_region.get('cookies') or {}
            self._session_ = requests.session()
            self._session_.cookies = requests.cookies.cookiejar_from_dict(persisted_cookies)
            self._session_.cookies.clear_expired_cookies()
        return self._session_

    def login(self, username, password):
        """ Authenticate LeetCode via username and password, and get session
        """
        if self.is_authenticated():
            logger.info('Already authenticated.')
            return

        # We need to get the csrftoken cookie
        del self._session.cookies['csrftoken']
        _ = self._session.get(_Endpoints.LOGIN)
        auth_result = self._session.post(_Endpoints.LOGIN, data={
            'login': username,
            'password': password,
            'remember': 'on',
            'csrfmiddlewaretoken': self._session.cookies.get('csrftoken')
        }, headers={
            'referer': _Endpoints.LOGIN
        })

        if not auth_result.ok:
            raise RuntimeError('Login failed: HTTP {}'.format(auth_result.status_code))
        logger.info('Login as {} succeed'.format(username))

        # Persist the authentication information to cache
        file_cache_region.set('cookies', self._session.cookies.get_dict())

    def logout(self):
        """ Logout of LeetCode """
        if not self.is_authenticated():
            logger.warning('Cannot logout, not authenticated')
            return

        self._session.get(_Endpoints.LOGOUT)
        file_cache_region.delete('cookies')

    def is_authenticated(self) -> bool:
        """ Check if we have already login """
        # We try to open submissions log. This is only accessible when you are logged in.
        submissions = self._session.get(_Endpoints.SUBMISSIONS, allow_redirects=False)
        if 'Location' in submissions.headers:
            # Redirect to login page, not authenticated
            logger.debug('Redirect to: {}'.format(submissions.headers['Location']))
            return False
        return True

    # NOTE: I cannot use -> dict to specify the return type, because
    # dogpile.cache does not support this. See:
    #   https://bitbucket.org/zzzeek/dogpile.cache/issues/96/support-python3-keyword-only-arguments-for
    def _all_problems(self):
        """ Collect all problems/user information """
        # LeetCode API is poorly designed, everything in the same endpoint...
        all_problems_result = self._session.get(_Endpoints.ALL_PROBLEMS)
        if not all_problems_result.ok:
            raise RuntimeError('Unable to retrieve api/problems/all, HTTP {}'.format(all_problems_result.status_code))
        return all_problems_result.json()

    def user_information(self) -> _UserInformation:
        if not self.is_authenticated():
            raise RuntimeError('Need login first')
        all_problems = self._all_problems()
        return _UserInformation(
            name=all_problems['user_name'],
            num_solved=all_problems['num_solved'],
            ac_easy=all_problems['ac_easy'],
            ac_medium=all_problems['ac_medium'],
            ac_hard=all_problems['ac_hard'],
            subscribed=all_problems['is_paid']
        )

    def problems_information(self) -> _AllProblemsInformation:
        all_problems = self._all_problems()
        if 'num_total' not in all_problems:
            raise RuntimeError('Unable to retrieve problem list.')
        level_problem_count = [
            len([i for i in all_problems['stat_status_pairs'] if i['difficulty']['level'] == difficulty])
            for difficulty in (1, 2, 3)
        ]
        return _AllProblemsInformation(
            total_num=all_problems['num_total'],
            num_easy=level_problem_count[0],
            num_medium=level_problem_count[1],
            num_hard=level_problem_count[2]
        )

    @memory_cache_region.cache_on_arguments()
    # NOTE: See NOTE for self._all_problems
    def problems(self):
        all_problems = self._all_problems()
        if 'stat_status_pairs' not in all_problems:
            raise RuntimeError('Unable to retrieve problem list.')
        result = []
        for item in all_problems['stat_status_pairs']:
            result.append(_ProblemListItem(
                index=item['stat']['question_id'],
                title=item['stat']['question__title'],
                slug=item['stat']['question__title_slug'],
                difficulty=item['difficulty']['level'],
                statistical_info=_ProblemStatisticalInformation(
                    total_submitted=item['stat']['total_submitted'],
                    total_accepts=item['stat']['total_acs'],
                    frequency=item['frequency']
                ),
                paid_only=item['paid_only'],
                status=item['status']
            ))
        return result

    def _find_problem_item(self, index: int) -> _ProblemListItem:
        problem_item = None
        for item in self.problems():
            if item.index == index:
                problem_item = item
                break
        if not problem_item:
            raise IndexError('Invalid index: {}'.format(index))
        return problem_item

    def problem(self, index):
        problem_item = self._find_problem_item(index)
        problem_description = self._session.get(
            _Endpoints.problem_description(problem_item.slug),
            # If the problem requires SUBSCRIBE while you are not logged in, it will let you redirect
            allow_redirects=False
        )
        if problem_description.status_code != 200:
            raise RuntimeError('Unable to get problem {}. {}: HTTP {}'.format(
                problem_item.index, problem_item.slug, problem_description.status_code))

        return _parse_raw_problem_html(problem_description.content)

    @memory_cache_region.cache_on_arguments()
    # NOTE: See NOTE for self._all_problems
    def _generate_submit_code_header(self, slug):
        return {
            'Referer': _Endpoints.problem_description(slug),
            'x-csrftoken': self._session.cookies.get('csrftoken', domain=_Endpoints.HOST),
            'X-Requested-With': 'XMLHttpRequest'
        }

    def _submit_code(self, index: int, slug: str, endpoint: str, language: str, code: str, test_case: str) -> dict:
        json_data = {
            'data_input': test_case,
            'judge_type': 'large',
            'lang': language,
            'question_id': str(index),
            'test_mode': False,
            'typed_code': code
        }
        run_test_result = self._session.post(
            endpoint,
            json=json_data,
            headers=self._generate_submit_code_header(slug)
        )
        if run_test_result.status_code != 200:
            raise RuntimeError(
                'Run code failed, HTTP: {} {}'.format(run_test_result.status_code, run_test_result.text))

        return run_test_result.json()

    def _collect_submit_results(self, submit_ids: typing.List[str], slug: str,
                                progress_callback: typing.Callable[[typing.List[str]], None] = None) \
            -> typing.List[dict]:
        num_ids = len(submit_ids)

        result_received = [False] * num_ids
        result_json = [{}] * num_ids

        header = self._generate_submit_code_header(slug)

        while not all(result_received):
            for index, _id in enumerate(submit_ids):
                if result_received[index]:
                    continue
                status = self._session.get(
                    _Endpoints.check_run_code_status(_id),
                    headers=header
                )
                if not status.ok:
                    logger.warning('Unable to GET {}: HTTP {}'.format(_Endpoints.check_run_code_status(_id),
                                                                      status.status_code))
                result_json[index] = status.json()
                if 'run_success' in result_json[index]:
                    result_received[index] = True

            if progress_callback:
                progress_callback([result_json[index]['state'] for index in range(num_ids)])

            time.sleep(1)

        return result_json

    def _report_test_run_submit_failures(self, result_json: dict, test_case: str) \
            -> typing.Union[_TestRunResultFailure, None]:
        if result_json['run_success']:
            return None

        status_code = result_json['status_code']

        if status_code == 20:
            return _TestRunResultFailure(
                test_case=test_case,
                error='Compile error: {}'.format(result_json['compile_error']),
                code_output=[])
        if status_code == 15:
            return _TestRunResultFailure(
                test_case=test_case,
                error='Runtime error: {}'.format(result_json['runtime_error']),
                code_output=result_json['code_output'])
        return _TestRunResultFailure(
            test_case=test_case,
            error=_RUN_ERROR_STATUS_CODE.get(status_code, 'Unknown error code: {}'.format(status_code)),
            code_output=result_json['code_output']
        )

    def test_run(self, index: int, language: str, code: str, test_case: str = None,
                 test_run_status_callback: typing.Callable[[typing.List[str]], None] = None) \
            -> typing.Union[_TestRunResultSuccess, _TestRunResultFailure]:

        problem_item = self._find_problem_item(index)
        problem_description = self.problem(index)
        test_case = test_case or problem_description.sample_testcase

        test_run_status_json = self._submit_code(
            index=index,
            slug=problem_item.slug,
            endpoint=_Endpoints.run_code(problem_item.slug),
            language=language,
            code=code,
            test_case=test_case
        )
        if 'error' in test_run_status_json:
            return _TestRunResultFailure(test_case=test_case, error=test_run_status_json['error'], code_output=[])

        # LeetCode will run two processes, one generates the expected value, the other generates the result from
        # test run
        expected_json, actual_json = self._collect_submit_results(
            submit_ids=[test_run_status_json['interpret_expected_id'], test_run_status_json['interpret_id']],
            slug=problem_item.slug,
            progress_callback=test_run_status_callback
        )

        # We always assume the standard solution is working, i.e. expected_json is always successful
        failure = self._report_test_run_submit_failures(result_json=actual_json, test_case=test_case)
        if failure:
            return failure

        return _TestRunResultSuccess(
            test_case=test_case,
            expected=expected_json['code_answer'],
            expected_time=expected_json['status_runtime'],
            actual=actual_json['code_answer'],
            actual_time=actual_json['status_runtime'],
            code_output=actual_json['code_output']
        )

    def _report_submit_code_failures(self, result_json: dict) \
            -> typing.Union[_SubmitResultFailure, None]:
        if result_json['run_success']:
            return None

        status_code = result_json['status_code']

        # Compile error
        if status_code == 20:
            return _SubmitResultFailure(
                error='Compile error: {}'.format(result_json['compile_error']),
                stdout=[],
                last_failed_test_case=None
            )
        if status_code == 15:
            return _SubmitResultFailure(
                error='Runtime error: {}'.format(result_json['runtime_error']),
                stdout=result_json['std_output'],
                last_failed_test_case=result_json['last_testcase']
            )
        return _SubmitResultFailure(
            error=_RUN_ERROR_STATUS_CODE.get(status_code, 'Unknown error code: {}'.format(status_code)),
            stdout=result_json['std_output'],
            last_failed_test_case=result_json['last_testcase']
        )

    def submit_code(self, index: int, language: str, source_code: str,
                    submit_status_callback: typing.Callable[[typing.List[str]], None] = None) \
            -> typing.Union[_SubmitResultAccepted, _SubmitResultWrongAnswer, _SubmitResultFailure]:

        problem_item = self._find_problem_item(index)
        # We read the problem in order to get the CSRF token
        self.problem(index)

        submit_status_json = self._submit_code(
            index=index,
            slug=problem_item.slug,
            endpoint=_Endpoints.submit_code(problem_item.slug),
            language=language,
            code=source_code,
            test_case=''
        )
        if 'error' in submit_status_json:
            return _SubmitResultFailure(error=submit_status_json['error'])

        submit_result_json = self._collect_submit_results(
            submit_ids=[submit_status_json['submission_id']],
            slug=problem_item.slug,
            progress_callback=submit_status_callback
        )[0]

        failure = self._report_submit_code_failures(submit_result_json)
        if failure:
            return failure

        total_test_cases = submit_result_json['total_testcases']
        total_correct = submit_result_json['total_correct']
        if total_test_cases == total_correct:
            # Accepted
            return _SubmitResultAccepted(
                time=submit_result_json['status_runtime'],
                stdout=submit_result_json['std_output'],
                total_test_cases=total_test_cases
            )
        else:
            # Wrong answer
            return _SubmitResultWrongAnswer(
                total_test_cases=total_test_cases,
                total_correct=total_correct,
                stdout=submit_result_json['std_output'],
                last_failed_test_case=submit_result_json['input'],
                last_expected=submit_result_json['expected_output'],
                last_actual=submit_result_json['code_output'],
                test_results=submit_result_json['compare_result'],
            )


def _setup_logger(level=logging.INFO) -> None:
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(FORMATTER)

    logger.addHandler(handler)


def _setup_cache() -> None:
    if not os.path.exists(CONFIG_DIRECTORY):
        logger.warning('Config directory not exist, creating {}'.format(CONFIG_DIRECTORY))
        os.makedirs(CONFIG_DIRECTORY, mode=0o700, exist_ok=True)
    file_cache_region.configure(
        backend='dogpile.cache.dbm',
        expiration_time=86400,
        arguments={
            'filename': os.path.join(CONFIG_DIRECTORY, 'cache.dbm')
        }
    )


@click.command()
@click.argument("username")
@click.password_option(confirmation_prompt=False)
def login(username: str, password: str) -> None:
    if not username:
        raise RuntimeError('No user name provided')

    lc = LeetCode()
    if lc.is_authenticated():
        raise RuntimeError('Already logged in')

    lc.login(username, password)


@click.command()
def info() -> None:
    lc = LeetCode()
    user_info = lc.user_information()
    problems_info = lc.problems_information()
    cprint('{:30s}'.format('User name:'), 'green', end='')
    cprint(user_info.name, 'red', end='')
    cprint(' [{}]'.format(
        'SUBSCRIBED' if user_info.subscribed else 'NOT SUBSCRIBED'), 'blue'
    )
    cprint('{:30s}'.format('Number problems solved:'), 'green', end='')
    cprint('{}/{} ({:.2f}%)'.format(
        user_info.num_solved,
        problems_info.total_num,
        user_info.num_solved / problems_info.total_num * 100), 'red'
    )
    cprint('    {:26s}'.format('Easy'), 'green', end='')
    cprint('{}/{} ({:.2f}%)'.format(
        user_info.ac_easy,
        problems_info.num_easy,
        user_info.ac_easy / problems_info.num_easy * 100), 'red'
    )
    cprint('    {:26s}'.format('Medium'), 'green', end='')
    cprint('{}/{} ({:.2f}%)'.format(
        user_info.ac_medium,
        problems_info.num_medium,
        user_info.ac_medium / problems_info.num_medium * 100), 'red'
    )
    cprint('    {:26s}'.format('Hard'), 'green', end='')
    cprint('{}/{} ({:.2f}%)'.format(
        user_info.ac_hard,
        problems_info.num_hard,
        user_info.ac_hard / problems_info.num_hard * 100), 'red'
    )


@click.command()
@click.argument('index')
@click.option('--language', default=None, help='Code defintion')
def problem(index: int, language: str) -> None:
    lc = LeetCode()
    # click is not respecting Python 3 type spec, have to cast
    problem = lc.problem(int(index))
    if language and language not in problem.code_definition:
        raise ValueError('Unsupported language {}, supported languages: {}'.format(
            language, problem.code_definition.keys()))

    print()
    cprint('  {}. {}'.format(problem.index, problem.title), 'white', attrs=['bold'])
    print()
    cprint(problem.description, 'white')
    print()

    if not language:
        if 'cpp' in problem.code_definition:
            language = 'cpp'
        else:
            language = next(iter(problem.code_definition.keys()))

    cprint('  Code definition:', 'white', attrs=['bold'], end='')
    cprint(' ({})'.format(language), 'yellow')
    print()
    cprint(problem.code_definition[language], 'white')
    print()

    cprint('  Default Test Sample:', 'white', attrs=['bold'])
    print()
    cprint(problem.sample_testcase, 'white')
    print()


def _prepare_post_code(index: str, source: str, language: str) -> typing.Tuple[int, str, str]:
    # From command line, index is string, we have to do the casting
    index = int(index)

    with open(source, 'r') as stream:
        source_code = stream.read()
    if not language:
        _, ext = os.path.splitext(source)
        if ext not in _LANGUAGE_EXT_MAPPER:
            raise KeyError(
                'Unrecognized extension {}, known extensions {}'.format(ext, _LANGUAGE_EXT_MAPPER.keys()))
        language = _LANGUAGE_EXT_MAPPER[ext]

    return (index, source_code, language)


@click.command()
@click.argument('index')
@click.argument('source')
@click.option('--language', default=None, help='Language')
@click.option('--test-input', default=None, help='Custom test input')
# FIXME mutual exclusive test-input and test-input-file
# @click.option('--test-input-file', default=None, help='Custom test input, read from file')
def test_run(index: int, source: str, language: str,
             test_input: str) -> None:  # Click does not respect Python3 type spec
    index, source_code, language = _prepare_post_code(index, source, language)

    lc = LeetCode()
    if not lc.is_authenticated():
        raise RuntimeError('Test run requires login.')

    cprint('  Testing problem {}'.format(index), 'white', attrs=['bold'])
    print()

    result = lc.test_run(
        index=index,
        language=language,
        code=source_code,
        test_case=test_input
    )

    cprint(' Test case:', 'blue', attrs=['bold'])
    print()
    print(result.test_case)
    print()

    cprint(' Code output:', 'blue', attrs=['bold'])
    print()
    for line in result.code_output:
        print(line)
    print()

    if isinstance(result, _TestRunResultFailure):
        cprint(' ERROR: ', 'red', attrs=['bold'], end='')
        cprint(result.error, 'red')
        print()
        return

    cprint(' Expected: ', 'green', attrs=['bold'], end='')
    cprint(result.expected_time, 'white')
    print()
    for line in result.expected:
        cprint(line, 'blue')
    print()
    cprint(' Actual: ', 'green', attrs=['bold'], end='')
    cprint(result.actual_time, 'white')
    print()
    for line in result.actual:
        cprint(line, 'blue')
    print()


@click.command()
@click.argument('index')
@click.argument('source')
@click.option('--language', default=None, help='Language')
def submit(index: int, source: str, language: str) -> None:
    index, source_code, language = _prepare_post_code(index, source, language)

    lc = LeetCode()
    if not lc.is_authenticated():
        raise RuntimeError('Test run requires login.')

    cprint('  Submit problem {}'.format(index), 'white', attrs=['bold'])
    print()

    result = lc.submit_code(index=index, language=language, source_code=source_code)

    cprint('  STDOUT:', 'blue', attrs=['bold'])
    print()
    print(result.stdout)
    print()

    if isinstance(result, _SubmitResultAccepted):
        cprint('  ACCEPTED ', 'green', attrs=['bold'], end='')
        cprint(result.time, 'white')
        print()
        cprint('  Passed {} test cases.'.format(result.total_test_cases), 'blue', attrs=['bold'])
        return

    if isinstance(result, _SubmitResultFailure):
        cprint(' ERROR: ', 'red', attrs=['bold'], end='')
        cprint(result.error, 'red')
        print()

    elif isinstance(result, _SubmitResultWrongAnswer):
        cprint('  TEST CASES: ', 'blue', attrs=['bold'], end='')
        cprint(result.total_correct, 'red', attrs=['bold'], end='')
        print('/', end='')
        cprint(result.total_test_cases, 'red', end='')
        print()
        for index, ch in enumerate(result.test_results):
            if index % 40 == 0:
                print('\n    ', end='')
            if ch == '0':
                cprint('F', 'red', end='')
            else:
                cprint('.', 'green', end='')
        print()
        print()

        cprint('  WRONG ANSWER', 'red', attrs=['bold'])
        print()
        cprint('  Expected:', 'red')
        print()
        print(result.last_expected)
        print()
        cprint('  Actual:', 'red')
        print()
        print(result.last_actual)
        print()

    if result.last_failed_test_case:
        cprint('  Last failed test case:', 'red')
        print()
        print(result.last_failed_test_case)
        print()


@click.command()
@click.option('--difficulty', help='Difficulty (easy/medium/hard)')
@click.option('--status', help='Status (intacted/tried/accepted)')
@click.option('--sort', help='Sort key (index/difficulty/frequency)[-(asc/desc)]', default='index-asc')
def list(difficulty: int, status: str, sort: str) -> None:
    lc = LeetCode()
    problems = lc.problems()

    def _sort(problems: typing.List[_ProblemListItem]) -> typing.List[_ProblemListItem]:
        sort_keys = sort.lower().split('-')
        if len(sort_keys) == 1:
            sort_keys.append('asc')

        if sort_keys[0] not in ('index', 'difficulty', 'frequency') or sort_keys[1] not in ('asc', 'desc'):
            raise ValueError('Unexpected sort instruction: {}'.format(sort))

        if sort_keys[0] == 'index':
            problems.sort(key=lambda i: i.index, reverse=sort_keys[1] == 'desc')
        elif sort_keys[0] == 'difficulty':
            problems.sort(
                key=lambda i: i.statistical_info.total_accepts / i.statistical_info.total_submitted,
                # On the opposite of difficulty, high accept rate implies easier problem
                reverse=sort_keys[1] != 'desc'
            )
            problems.sort(key=lambda i: i.difficulty, reverse=sort_keys[1] == 'desc')
        elif sort_keys[0] == 'frequency':
            problems.sort(key=lambda i: i.frequency, reverse=sort_keys[1] == 'desc')

        return problems

    def _filter(problems: typing.List[_ProblemListItem]) -> typing.Generator[_ProblemListItem, None, None]:
        STATUS_MAPPER = {'intacted': '-', 'tried': '?', 'accepted': '*'}
        if status and not lc.is_authenticated():
            raise RuntimeError('Need login to see problem status')
        for item in problems:
            if difficulty and item.difficulty != {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]:
                continue
            if status and _STATUS_NAME_MAPPER[item.status] != STATUS_MAPPER[status]:
                continue
            yield item

    cprint(_LIST_PROBLEM_FORMATTER.format('', 'Id', 'Title', 'Difficulty', 'Rate', 'Status'), 'white',
           attrs=['bold'])
    cprint('=' * len(_LIST_PROBLEM_FORMATTER.format('', '', '', '', '', '')), 'white', attrs=['bold'])
    count = 0
    for item in _filter(_sort(problems)):
        count += 1
        cprint(_PrintFormatter.STATUS.format(_STATUS_NAME_MAPPER[item.status]),
               color=_STATUS_COLOR_MAPPER[item.status], end='')
        cprint(_PrintFormatter.INDEX.format(str(item.index)),
               color='white', end='')
        cprint(_PrintFormatter.TITLE.format(item.title),
               color='white', end='')
        cprint(_PrintFormatter.DIFFICULTY.format(_DIFFICULTY_NAME_MAPPER[item.difficulty]),
               color=_DIFFICULTY_COLOR_MAPPER[item.difficulty], end='')
        cprint(_PrintFormatter.ACCEPTANCE_RATE.format('{:.2f}%'.format(item.statistical_info.total_accepts /
                                                                       item.statistical_info.total_submitted * 100)),
               color='yellow', end='')
        cprint(_PrintFormatter.SUBSCRIBE_REQUIRED.format('SUBSCRIBE' if item.paid_only else ''),
               color='cyan', attrs=['bold'], end='')
        print()
    cprint('\n TOTAL PROBLEMS: ', 'white', end='')
    cprint('{:5d}'.format(count), 'red', attrs=['bold'])


@click.command()
def logout() -> None:
    LeetCode().logout()


@click.group()
def main() -> None:
    pass


main.add_command(login)
main.add_command(info)
main.add_command(list)
main.add_command(problem)
main.add_command(test_run)
main.add_command(submit)
main.add_command(logout)

if __name__ == '__main__':
    _setup_logger()
    _setup_cache()
    main()
