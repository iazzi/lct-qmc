#ifndef LOGGER_H
#define LOGGER_H

#include <pthread.h>
#include <iostream>

class Logger {
	std::ostream* m_stream;
	std::mutex m_mutex;
	char m_separator;
	public:
	Logger() : m_stream(&std::clog), m_separator(' ') { }
	Logger(std::ostream& o) : m_stream(&o), m_separator(' ') { }
	~Logger() { }
	std::ostream& stream() { return *m_stream; }
	std::ostream& stream(std::ostream& s) { m_stream = &s; return *m_stream; }
	void lock() { m_mutex.lock(); }
	void unlock() { m_mutex.unlock(); }
	char separator() { return m_separator; }
	void setSeparator(const char s) { m_separator = s; }
};

template <typename Helper, typename T>
class LoggerHelperBase {
	protected:
		Helper& m_helper;
		const T& m_value;
		bool m_active;
	public:
		LoggerHelperBase(Helper& helper, const T& value)
			: m_helper(helper)
			  , m_value(value)
			  , m_active(true) { }

		void deactivate() { m_active = false; }
};

template <typename Helper, typename T>
class LoggerHelper : public LoggerHelperBase<Helper, T> {
	public:
		typedef LoggerHelperBase<Helper, T> B;

		LoggerHelper(Helper& helper, const T& value)
			: LoggerHelperBase<Helper, T>(helper, value) {
				helper.deactivate();
			}

		~LoggerHelper() {
			if (B::m_active) {
				Logger& logger = getLogger();
				logger.lock();
				run(logger.stream(), logger.separator());
				logger.stream() << std::endl;
				logger.unlock();
			}
		}

		void run(std::ostream& stream, const char s) {
			B::m_helper.run(stream, s);
			stream << s << B::m_value;
		}

		Logger& getLogger() { return B::m_helper.getLogger(); }
};

template <typename T>
class LoggerHelper<Logger, T> : public LoggerHelperBase<Logger, T> {
	public:
		typedef LoggerHelperBase<Logger, T> B;

		LoggerHelper(Logger& logger, const T& value)
			: LoggerHelperBase<Logger, T>(logger, value) { }

		~LoggerHelper() {
			if (B::m_active) {
				Logger& logger = B::m_helper;
				logger.lock();
				run(logger.stream(), ' ');
				logger.stream() << std::endl;
				logger.unlock();
			}
		}

		void run(std::ostream& stream, const char s) {
			stream << B::m_value;
		}

		Logger& getLogger() { return B::m_helper; }
};

template <typename T>
LoggerHelper<Logger, T> operator<<(Logger& logger, const T& value) {
	return LoggerHelper<Logger, T>(logger, value);
}

template <typename Helper, typename T1, typename T2>
LoggerHelper<LoggerHelper<Helper, T1>, T2> operator<<(LoggerHelper<Helper, T1> helper, const T2& value) {
	return LoggerHelper<LoggerHelper<Helper, T1>, T2>(helper, value);
}


#endif // LOGGER_H


